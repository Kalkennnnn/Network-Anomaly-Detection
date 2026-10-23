"""
Модуль захвата трафика и извлечения признаков
capture.py
"""

import os
import time
import threading
import numpy as np
from scapy.all import sniff, IP, TCP, UDP, PcapReader

# ── Настройки ──────────────────────────────────────────────────────────────────
FLOW_TIMEOUT     = 5     # секунд без пакетов → поток завершён
CLEANUP_INTERVAL = 2     # как часто уборщик проверяет потоки, с
MIN_PACKETS      = 2     # минимум пакетов для классификации
MAX_PACKETS      = 20    # принудительный сброс потока после N пакетов
EARLY_FLUSH_PKTS = 5     # ранний предварительный показ после N пакетов

SIMULATE_ATTACK  = False  # True → каждый 5-й поток подменяется вектором атаки из датасета


_ATTACK_VECTORS = [
    # DoS флудинг (нулевой обратный трафик, малые паузы)
    np.array([0.0,    0.0,      0.0,       0.0,   443.64,
              0.0,    0.0,      0.0,       994.0,  192.0], dtype=np.float32),
    # DDoS (высокий Bwd Std, огромный Flow IAT Max)
    np.array([2189.77, 833.07, 2667024.41, 11601.0, 26300000.0,
              56.0,    20.0,   7.0,         70400000.0, 172.0], dtype=np.float32),
    # PortScan (малые пакеты, нет ответа)
    np.array([0.0,    5.0,     5.33,       6.0,   0.0,
              2.0,    2.0,     2.0,         30.0,   24.0], dtype=np.float32),
    # Slowloris (огромные паузы)
    np.array([1762.27, 1163.30, 2557770.67, 11607.0, 475.18,
              26.0,    20.0,    8.67,        108823.0, 72.0], dtype=np.float32),
]

FEATURE_NAMES = [
    "Bwd Packet Length Std",
    "Average Packet Size",
    "Packet Length Variance",
    "Total Length of Bwd Packets",
    "Fwd IAT Std",
    "Total Length of Fwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Mean",
    "Flow IAT Max",
    "Fwd Header Length",
]



# ══════════════════════════════════════════════════════════════════════════════
#  Flow — один сетевой поток
# ══════════════════════════════════════════════════════════════════════════════

class Flow:
    __slots__ = (
        "flow_id", "start_time", "last_time",
        "fwd_lengths", "fwd_times", "fwd_header_lengths",
        "bwd_lengths", "all_times", "early_flushed",
    )

    def __init__(self, flow_id: tuple, timestamp: float):
        self.flow_id              = flow_id
        self.start_time           = timestamp
        self.last_time            = timestamp
        self.fwd_lengths:   list  = []
        self.fwd_times:     list  = []
        self.fwd_header_lengths: list = []
        self.bwd_lengths:   list  = []
        self.all_times:     list  = []
        self.early_flushed: bool  = False

    def add_packet(self, length: int, header_length: int,
                   timestamp: float, is_forward: bool) -> None:
        self.last_time = timestamp
        self.all_times.append(timestamp)
        if is_forward:
            self.fwd_lengths.append(length)
            self.fwd_times.append(timestamp)
            self.fwd_header_lengths.append(header_length)
        else:
            self.bwd_lengths.append(length)

    @property
    def total_packets(self) -> int:
        return len(self.fwd_lengths) + len(self.bwd_lengths)

    def is_expired(self, now: float, timeout: float = FLOW_TIMEOUT) -> bool:
        return (now - self.last_time) > timeout

    def is_full(self) -> bool:
        return self.total_packets >= MAX_PACKETS

    def extract_features(self) -> "np.ndarray | None":
        """
        Возвращает вектор из 10 признаков или None если данных недостаточно.
        Временны́е интервалы (IAT) переводятся в микросекунды (× 1e6),
        чтобы совпадать с масштабом обучающей выборки CIC-IDS2017.
        Пакеты всегда приходят в хронологическом порядке, поэтому
        all_times не сортируется повторно.
        """
        if len(self.fwd_lengths) < 2:
            return None

        all_lengths = self.fwd_lengths + self.bwd_lengths

        bwd_std   = float(np.std(self.bwd_lengths)) if self.bwd_lengths else 0.0
        avg_size  = float(np.mean(all_lengths))
        pkt_var   = float(np.var(all_lengths))
        total_bwd = float(sum(self.bwd_lengths))
        total_fwd = float(sum(self.fwd_lengths))
        fwd_max   = float(max(self.fwd_lengths))
        fwd_mean  = float(np.mean(self.fwd_lengths))

        # Fwd IAT Std (мкс)
        if len(self.fwd_times) >= 2:
            fwd_iats    = [self.fwd_times[i + 1] - self.fwd_times[i]
                           for i in range(len(self.fwd_times) - 1)]
            fwd_iat_std = float(np.std(fwd_iats)) * 1e6
        else:
            fwd_iat_std = 0.0

        # Flow IAT Max (мкс) — пакеты уже упорядочены, сортировка не нужна
        if len(self.all_times) >= 2:
            all_iats     = [self.all_times[i + 1] - self.all_times[i]
                            for i in range(len(self.all_times) - 1)]
            flow_iat_max = float(max(all_iats)) * 1e6
        else:
            flow_iat_max = 0.0

        fwd_header_len = float(sum(self.fwd_header_lengths))

        return np.array([
            bwd_std, avg_size, pkt_var, total_bwd,
            fwd_iat_std, total_fwd, fwd_max, fwd_mean,
            flow_iat_max, fwd_header_len,
        ], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  Вспомогательные функции — общие для FlowCapture и PcapCapture
# ══════════════════════════════════════════════════════════════════════════════

def _get_flow_id(pkt) -> "tuple | None":
    """Возвращает 5-tuple (src_ip, src_port, dst_ip, dst_port, proto) или None."""
    if IP not in pkt:
        return None
    src_ip, dst_ip = pkt[IP].src, pkt[IP].dst
    proto = pkt[IP].proto
    if TCP in pkt:
        src_port, dst_port = pkt[TCP].sport, pkt[TCP].dport
    elif UDP in pkt:
        src_port, dst_port = pkt[UDP].sport, pkt[UDP].dport
    else:
        src_port = dst_port = 0
    return (src_ip, src_port, dst_ip, dst_port, proto)


def _get_header_length(pkt) -> int:
    """Суммарная длина IP + транспортного заголовка в байтах."""
    ip_hdr = pkt[IP].ihl * 4 if IP in pkt else 20
    if TCP in pkt:
        return ip_hdr + pkt[TCP].dataofs * 4
    if UDP in pkt:
        return ip_hdr + 8
    return ip_hdr


# ══════════════════════════════════════════════════════════════════════════════
#  FlowCapture — онлайн-захват с сетевого интерфейса
# ══════════════════════════════════════════════════════════════════════════════

class FlowCapture:
    def __init__(self, iface: str, on_flow_ready,
                 flow_timeout: float = FLOW_TIMEOUT,
                 min_packets: int    = MIN_PACKETS):
        self.iface         = iface
        self.on_flow_ready = on_flow_ready
        self.flow_timeout  = flow_timeout
        self.min_packets   = min_packets

        self.flows: dict   = {}
        self.lock          = threading.Lock()
        self.running       = False
        self._thread       = None
        self._cleaner      = None
        self._flow_counter = 0
        self._counter_lock = threading.Lock()

    # ── приватные ─────────────────────────────────────────────────────────────

    def _send_flow(self, fid: tuple, flow: Flow) -> None:
        if flow.total_packets < self.min_packets:
            return

        if SIMULATE_ATTACK:
            with self._counter_lock:
                self._flow_counter += 1
                n = self._flow_counter
            if n % 5 == 0:
                # Каждый 5-й поток подменяем вектором атаки из датасета
                idx = (n // 5 - 1) % len(_ATTACK_VECTORS)
                attack_vec = _ATTACK_VECTORS[idx]
                print(f"[SimAttack] Поток #{n} → атака тип {idx + 1}")
                self.on_flow_ready(fid, attack_vec)
                return

        features = flow.extract_features()
        if features is None:
            return
        self.on_flow_ready(fid, features)

    def _process_packet(self, pkt) -> None:
        flow_id = _get_flow_id(pkt)
        if flow_id is None:
            return

        rev_id    = (flow_id[2], flow_id[3], flow_id[0], flow_id[1], flow_id[4])
        canonical = min(flow_id, rev_id)
        now       = time.time()
        pkt_len   = len(pkt)
        hdr_len   = _get_header_length(pkt)
        is_fwd    = (flow_id == canonical)

        closed_flow    = None
        do_early_flush = False
        early_features = None

        with self.lock:
            if canonical not in self.flows:
                self.flows[canonical] = Flow(canonical, now)
            flow = self.flows[canonical]
            flow.add_packet(pkt_len, hdr_len, now, is_fwd)

            if flow.is_full():
                # Закрываем поток; extract_features вызовем вне лока
                closed_flow = flow
                del self.flows[canonical]
            elif flow.total_packets == EARLY_FLUSH_PKTS and not flow.early_flushed:
                # Снимаем признаки прямо под локом → нет race condition
                flow.early_flushed = True
                early_features = flow.extract_features()
                do_early_flush = True

        # Тяжёлые операции — вне лока
        if closed_flow is not None:
            self._send_flow(canonical, closed_flow)
        elif do_early_flush and early_features is not None:
            self.on_flow_ready(canonical, early_features)

    def _cleanup_expired(self) -> None:
        while self.running:
            time.sleep(CLEANUP_INTERVAL)
            now     = time.time()
            expired = []
            with self.lock:
                for fid, flow in list(self.flows.items()):
                    if flow.is_expired(now, self.flow_timeout):
                        expired.append((fid, flow))
                        del self.flows[fid]
            for fid, flow in expired:
                self._send_flow(fid, flow)

    # ── публичный API ──────────────────────────────────────────────────────────

    def start(self) -> None:
        self.running  = True
        self._cleaner = threading.Thread(
            target=self._cleanup_expired, daemon=True)
        self._cleaner.start()
        self._thread  = threading.Thread(
            target=lambda: sniff(
                iface=self.iface,
                prn=self._process_packet,
                store=False,
                stop_filter=lambda _: not self.running,
            ),
            daemon=True,
        )
        self._thread.start()
        print(f"[FlowCapture] Запущен: {self.iface} | "
              f"timeout={self.flow_timeout}с | max_packets={MAX_PACKETS} | "
              f"simulate_attack={SIMULATE_ATTACK}")

    def stop(self) -> None:
        self.running = False
        print("[FlowCapture] Остановлен.")

    def flush(self) -> None:
        """Принудительно сбрасывает все текущие потоки (вызывать перед stop)."""
        with self.lock:
            flows_copy = list(self.flows.items())
            self.flows.clear()
        for fid, flow in flows_copy:
            self._send_flow(fid, flow)


# ══════════════════════════════════════════════════════════════════════════════
#  PcapCapture — офлайн-анализ .pcap / .pcapng файла
# ══════════════════════════════════════════════════════════════════════════════

class PcapCapture:
    """
    Читает pcap-файл и классифицирует потоки.

    Отличия от FlowCapture:
    • Временны́е метки берутся из заголовков пакетов — IAT корректны.
    • Прогресс считается по позиции в файле — без двойного прохода по файлу.
    • По завершении автоматически сбрасывает незакрытые потоки.
    • Коллбэки: on_progress(pct: int), on_done(total_flows: int).
    """

    def __init__(self, filepath: str, on_flow_ready,
                 flow_timeout: float = FLOW_TIMEOUT,
                 min_packets: int    = MIN_PACKETS,
                 on_progress=None,
                 on_done=None):
        self.filepath      = filepath
        self.on_flow_ready = on_flow_ready
        self.flow_timeout  = flow_timeout
        self.min_packets   = min_packets
        self.on_progress   = on_progress
        self.on_done       = on_done

        self.flows: dict   = {}
        self._flow_counter = 0
        self._thread       = None
        self.running       = False

    # ── приватные ─────────────────────────────────────────────────────────────

    def _send_flow(self, fid: tuple, flow: Flow) -> None:
        if flow.total_packets < self.min_packets:
            return

        if SIMULATE_ATTACK:
            with self._counter_lock:
                self._flow_counter += 1
                n = self._flow_counter
            if n % 5 == 0:
                # Каждый 5-й поток подменяем вектором атаки из датасета
                idx = (n // 5 - 1) % len(_ATTACK_VECTORS)
                attack_vec = _ATTACK_VECTORS[idx]
                print(f"[SimAttack] Поток #{n} → атака тип {idx + 1}")
                self.on_flow_ready(fid, attack_vec)
                return

        features = flow.extract_features()
        if features is None:
            return
        self.on_flow_ready(fid, features)
        self._flow_counter += 1

    def _process_packet(self, pkt, pkt_time: float) -> None:
        flow_id = _get_flow_id(pkt)
        if flow_id is None:
            return

        rev_id    = (flow_id[2], flow_id[3], flow_id[0], flow_id[1], flow_id[4])
        canonical = min(flow_id, rev_id)
        pkt_len   = len(pkt)
        hdr_len   = _get_header_length(pkt)
        is_fwd    = (flow_id == canonical)

        # Таймаут по реальному времени из заголовка pcap
        if canonical in self.flows:
            existing = self.flows[canonical]
            if (pkt_time - existing.last_time) > self.flow_timeout:
                self._send_flow(canonical, existing)
                del self.flows[canonical]

        if canonical not in self.flows:
            self.flows[canonical] = Flow(canonical, pkt_time)

        flow = self.flows[canonical]
        flow.add_packet(pkt_len, hdr_len, pkt_time, is_fwd)

        if flow.total_packets >= MAX_PACKETS:
            self._send_flow(canonical, flow)
            del self.flows[canonical]

    def _run(self) -> None:
        print(f"[PcapCapture] Открываю файл: {self.filepath}")
        try:
            # Прогресс по позиции в файле — один проход, без двойного чтения
            file_size = os.path.getsize(self.filepath)
            last_pct  = -1

            with PcapReader(self.filepath) as reader:
                for pkt in reader:
                    if not self.running:
                        break

                    try:
                        pkt_time = float(pkt.time)
                    except Exception:
                        pkt_time = time.time()

                    self._process_packet(pkt, pkt_time)

                    if file_size > 0 and self.on_progress:
                        try:
                            pct = min(int(reader.f.tell() / file_size * 100), 99)
                            if pct != last_pct:
                                self.on_progress(pct)
                                last_pct = pct
                        except Exception:
                            pass  # tell() недоступен — пропускаем молча

            # Сбрасываем незакрытые потоки
            if self.running:
                for fid, flow in list(self.flows.items()):
                    self._send_flow(fid, flow)
                self.flows.clear()
                if self.on_progress:
                    self.on_progress(100)

        except Exception as e:
            print(f"[PcapCapture] Ошибка: {e}")
        finally:
            self.running = False
            total = self._flow_counter
            print(f"[PcapCapture] Завершено. Потоков передано: {total}")
            if self.on_done:
                self.on_done(total)

    # ── публичный API ──────────────────────────────────────────────────────────

    def start(self) -> None:
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.running = False
        print("[PcapCapture] Остановлен досрочно.")
