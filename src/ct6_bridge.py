# -*- coding: utf-8 -*-
"""simulate_ct6 的 Agent 桥接口（TCP + 发现文件）。

设计沿用本工作区既有的 MCP 桥模式（同 ssd_vr）：
  * 应用内起一个 TCP 服务线程，监听 127.0.0.1 的随机端口
  * 端口写入发现文件 `I:\\dsw\\ct6_bridge.json`，Agent 侧读该文件即可连上
  * 协议：按行分隔的 JSON，请求 {"op": "...", ...} → 响应 {"ok": bool, ...}

**线程安全**：Qt 控件只能在主线程操作，因此所有 op 都通过
`call_main()`（QTimer.singleShot + Event）投递到主线程执行并同步取回结果。

变量空间（程序化导出的真实清单）
--------------------------------
输入（43 个）
  A 几何滑块 19：pitch, scan_length, slice_thickness, slice_interval, shots_per_source,
                alpha, RA, RB, FDD, SFOV_A, SFOV_B, Z_coverage, rotation_time,
                sampling_rate, pixel_xy, pixel_z, n_ch_set, bowtie_sfov, bowtie_edge
  B 开关/下拉 4：nch_auto_sw, asymmetric_cb, arch, scan_mode
  C FBP 控件 12：src, path, kernel, matrix, scan(360/短扫描), parker, rebin,
                helical(z 插值), lowpass, cone(128排锥束), force_arch, play
  D Fermi 8：material, thickness, voltage, temp_k, energy_keV, kvp, n_bins, bin_mode
输出（121 个）：几何/架构/阵列/协议/剂量/模态等派生量，见 results

op 一览
-------
  ping                                  连通性
  list                                  变量清单（含取值范围/类型）
  get                                  全部输入当前值 + 全部派生结果
  set   {"params":{...}, "settle_ms":0}  按名写变量并触发重算
  arch  {"arch":"pcct", "settle_ms":0}   切 CT 架构
  scan  {"mode":"helical"}               切扫描模式
  fbp   {"settle_ms":0}                  执行 FBP 重建
  fermi {"settle_ms":0}                  执行 Fermi 响应计算
  view  {"preset":"axial"}               3D 视角（coronal/sagittal/axial/three_quarter…）
  shot  {}                               窗口截图（PNG base64）
"""

import base64
import json
import os
import socket
import threading
import time

DISCOVERY = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ct6_bridge.json')

ARCH_KEYS = {'dual_source': '双源', 'single_wide': '单源宽体', 'dual_layer': '双层',
             'pcct': '光子计数', 'static_multi': '静态多源', 'synchrotron': '同步辐射'}
# 下拉框完整选项文本（setCurrentText 必须精确匹配，否则静默失败）
ARCH_FULL = {
    'dual_source': 'CT 架构：双源 (Dual-Source)',
    'single_wide': 'CT 架构：单源宽体 (Single-Source Wide-Body)',
    'dual_layer': 'CT 架构：双层探测器 (Dual-Layer Spectral)',
    'pcct': 'CT 架构：光子计数 (Photon-Counting 8-bin)',
    'static_multi': 'CT 架构：静态多源 (Stationary 24-Source)',
    'synchrotron': 'CT 架构：同步辐射 (Synchrotron · 单元光子计数改造)'}
SCAN_FULL = {'axial': '扫描模式：轴扫 (Axial)', 'helical': '扫描模式：螺旋 (Helical)',
             'static': '扫描模式：静态 (Static, 不旋转)'}


def _inventory(win):
    """输入变量清单：名称 → {kind, value, min, max, values}。"""
    out = {}
    # 同步辐射仿真参数只在切到该架构时列出（与 Fermi 页签同一口径），
    # 免得 Agent 在别的架构下改这些不起作用的量还以为生效了。
    sync_on = getattr(win, 'arch_key', None) == 'synchrotron'
    for k, s in getattr(win, 'sliders', {}).items():
        if k.startswith('sync_') and not sync_on:
            continue
        try:
            v = s.value()
        except Exception:
            v = None
        dec = int(getattr(s, 'decimals', 0) or 0)
        sc = 10 ** dec
        item = {'kind': 'slider', 'value': v, 'decimals': dec}
        try:
            sb = s.slider()
            # 注意：QSlider 以 value×10^decimals 的整数存储 → min/max 需换算回**真实单位**，
            # 否则调用方按 min/max 取值再写回会溢出钳位（静默失效）。
            item['min'] = sb.minimum() / sc
            item['max'] = sb.maximum() / sc
            item['step'] = sb.singleStep() / sc
        except Exception:
            pass
        out[k] = item
    for nm in ('nch_auto_sw', 'asymmetric_cb'):
        o = getattr(win, nm, None)
        if o is not None:
            try:
                out[nm] = {'kind': 'switch', 'value': bool(o.isChecked())}
            except Exception:
                pass
    if sync_on:
        for name, attr in (('sync_source', 'sync_source_combo'),
                           ('sync_detector', 'sync_det_combo')):
            c = getattr(win, attr, None)
            if c is not None:
                try:
                    out[name] = {'kind': 'combo', 'value': c.combo_box().currentText()}
                except Exception:
                    pass
        o = getattr(win, 'sync_phase_sw', None)
        if o is not None:
            try:
                out['sync_phase_sw'] = {'kind': 'switch', 'value': bool(o.isChecked())}
            except Exception:
                pass
        o = getattr(win, 'sync_deconv_sw', None)
        if o is not None:
            try:
                out['sync_deconv_sw'] = {'kind': 'switch', 'value': bool(o.isChecked())}
            except Exception:
                pass
    try:
        out['arch'] = {'kind': 'combo', 'value': win.arch_key,
                       'values': list(ARCH_KEYS.keys()),
                       'text': win.arch_combo.combo_box().currentText()}
    except Exception:
        pass
    try:
        out['scan_mode'] = {'kind': 'combo',
                            'value': win.scan_mode_combo.combo_box().currentText()}
    except Exception:
        pass
    p = getattr(getattr(win, 'recon_widget', None), 'fbp_panel', None)
    if p is not None:
        for nm in ('src_combo', 'kernel_combo', 'matrix_combo', 'scan_combo',
                   'helical_combo'):
            c = getattr(p, nm, None)
            if c is not None:
                try:
                    out[nm.replace('_combo', '')] = {
                        'kind': 'combo', 'value': c.combo_box().currentText()}
                except Exception:
                    pass
        for nm in ('parker_sw', 'rebin_sw', 'cone_sw', 'force_arch_sw', 'play_sw'):
            c = getattr(p, nm, None)
            if c is not None:
                try:
                    out[nm.replace('_sw', '')] = {'kind': 'switch',
                                                  'value': bool(c.isChecked())}
                except Exception:
                    pass
        try:
            out['lowpass'] = {'kind': 'slider', 'value': float(p.lowpass_slider.value())}
        except Exception:
            pass
    try:
        out['ring_seg'] = {'kind': 'combo',
                           'value': win.ring_combo.combo_box().currentText(),
                           'values': ['采集环段：全环 24 源 (360°)',
                                      '采集环段：短扫描 16 源 (180°+扇角)',
                                      '采集环段：半环 12 源 (180°)']}
    except Exception:
        pass
    _rw = getattr(win, 'recon_widget', None)
    fp = getattr(_rw, 'tab_fermi', None)
    # Fermi 页仅在光子计数架构下可见；不可见时不列举其变量（否则值为 None 会误导调用方）
    if fp is not None and _rw is not None and _rw.tabs.indexOf(fp) >= 0:
        for nm, key in (('mat_combo', 'f_material'), ('bins_combo', 'f_n_bins'),
                        ('mode_combo', 'f_bin_mode')):
            c = getattr(fp, nm, None)
            if c is not None:
                try:
                    out[key] = {'kind': 'combo', 'value': c.combo_box().currentText()}
                except Exception:
                    pass
        for nm, key in (('s_thick', 'f_thickness'), ('s_bias', 'f_voltage'),
                        ('s_temp', 'f_temp_k'), ('s_energy', 'f_energy'),
                        ('s_kvp', 'f_kvp')):
            c = getattr(fp, nm, None)
            if c is not None:
                try:
                    out[key] = {'kind': 'slider', 'value': float(c.value())}
                except Exception:
                    pass
    return out


def _results(win):
    """全部派生结果（121 键）+ FBP/静态状态。"""
    p = getattr(getattr(win, 'recon_widget', None), 'fbp_panel', None)
    r = dict(getattr(p, 'results', None) or {})
    extra = {
        'arch_key': getattr(win, 'arch_key', None),
        'gear': {'tube': getattr(win, 'gear_tag', None)},
        'scan_mode': (win.scan_mode_combo.combo_box().currentText()
                      if hasattr(win, 'scan_mode_combo') else None),
        'fbp_ready': bool(getattr(p, 'sino', None) is not None),
        'sino_shape': list(getattr(p, 'sino').shape) if getattr(p, 'sino', None) is not None else None,
        'leap_rows': None, 'leap_cols': None,
    }
    try:
        extra['leap_rows'] = int(p.engine.ct.get_numRows())
        extra['leap_cols'] = int(p.engine.ct.get_numCols())
        extra['leap_angles'] = int(p.engine.ct.get_numAngles())
    except Exception:
        pass
    r.update(extra)
    r['inputs'] = {k: v.get('value') if isinstance(v, dict) else v
                   for k, v in _inventory(win).items()}
    return r


def _apply(win, params):
    """按名写变量（滑块/开关/下拉），返回逐项结果。"""
    rep = {}
    for k, v in (params or {}).items():
        try:
            if k in getattr(win, 'sliders', {}):
                s = win.sliders[k]
                sb = s.slider()
                sb.setValue(int(round(float(v) * (10 ** getattr(s, 'decimals', 0)))))
                rep[k] = 'ok'
                continue
            if k in ('nch_auto_sw', 'asymmetric_cb'):
                getattr(win, k).setChecked(bool(v))
                rep[k] = 'ok'
                continue
            if k == 'arch':
                key = str(v)
                if key not in ARCH_FULL:      # 允许传完整文本
                    key = next((a for a, t in ARCH_FULL.items() if str(v) in t), None)
                if key is None:
                    rep[k] = 'unknown arch'
                    continue
                win.arch_combo.combo_box().setCurrentText(ARCH_FULL[key])
                rep[k] = 'ok'
                continue
            if k == 'scan_mode':
                key = str(v)
                if key not in SCAN_FULL:
                    key = ('helical' if str(v).startswith('h') or '螺旋' in str(v) else
                           'static' if str(v).startswith('s') or '静态' in str(v) else 'axial')
                win.scan_mode_combo.combo_box().setCurrentText(SCAN_FULL[key])
                rep[k] = 'ok'
                continue
            if k in ('ring_seg', 'ring'):
                txt = str(v)
                if '环段' not in txt:
                    txt = ('采集环段：半环 12 源 (180°)' if str(v).startswith('12')
                           else '采集环段：短扫描 16 源 (180°+扇角)' if str(v).startswith('16')
                           else '采集环段：全环 24 源 (360°)')
                win.ring_combo.combo_box().setCurrentText(txt)
                rep[k] = 'ok'
                continue
            if k in ('sync_source', 'sync_detector'):
                box = (win.sync_source_combo if k == 'sync_source'
                       else win.sync_det_combo)
                box.combo_box().setCurrentText(str(v))
                rep[k] = 'ok'
                continue
            if k in ('sync_phase_sw', 'sync_deconv_sw'):
                getattr(win, k).setChecked(bool(v))
                rep[k] = 'ok'
                continue
            p = win.recon_widget.fbp_panel
            cmap = {'src': 'src_combo', 'kernel': 'kernel_combo', 'matrix': 'matrix_combo',
                    'scan': 'scan_combo', 'helical': 'helical_combo'}
            if k in cmap:
                getattr(p, cmap[k]).combo_box().setCurrentText(str(v))
                rep[k] = 'ok'
                continue
            smap = {'parker': 'parker_sw', 'rebin': 'rebin_sw', 'cone': 'cone_sw',
                    'force_arch': 'force_arch_sw', 'play': 'play_sw'}
            if k in smap:
                getattr(p, smap[k]).setChecked(bool(v))
                rep[k] = 'ok'
                continue
            if k == 'lowpass':
                p.lowpass_slider.slider().setValue(int(round(float(v))))
                rep[k] = 'ok'
                continue
            fp = win.recon_widget.tab_fermi
            if fp is not None and k.startswith('f_'):
                fmap = {'f_material': 'mat_combo', 'f_n_bins': 'bins_combo',
                        'f_bin_mode': 'mode_combo'}
                if k in fmap:
                    getattr(fp, fmap[k]).combo_box().setCurrentText(str(v))
                else:
                    smap2 = {'f_thickness': 's_thick', 'f_voltage': 's_bias',
                             'f_temp_k': 's_temp', 'f_energy': 's_energy', 'f_kvp': 's_kvp'}
                    c = getattr(fp, smap2.get(k, ''), None)
                    if c is None:
                        rep[k] = 'unknown'
                        continue
                    c.slider().setValue(int(round(float(v) * (10 ** getattr(c, 'decimals', 0)))))
                rep[k] = 'ok'
                continue
            rep[k] = 'unknown'
        except Exception as e:
            rep[k] = f'error: {e}'
    return rep


from PySide6.QtCore import QObject, Signal, Qt


class Bridge(QObject):
    """在应用内提供 TCP+JSON 控制接口（QObject：用队列信号跨线程投递到主线程）。"""

    _sig = Signal(object, object)

    def __init__(self, win, host='127.0.0.1', port=0):
        super().__init__()
        self._sig.connect(self._invoke, Qt.ConnectionType.QueuedConnection)
        self.win = win
        self.host = host
        self.port = int(port)
        self._srv = None
        self._threads = []
        self._running = False

    # ---------- 主线程投递 ----------
    def _invoke(self, fn, payload):
        """在主线程执行（由队列信号触发）。"""
        box, ev = payload
        try:
            box['r'] = fn()
        except Exception as exc:
            box['e'] = f'{type(exc).__name__}: {exc}'
        finally:
            ev.set()

    def call_main(self, fn, timeout=180.0):
        """把 fn 投递到 Qt 主线程执行并同步取回结果（线程安全）。"""
        box, ev = {}, threading.Event()
        self._sig.emit(fn, (box, ev))
        if not ev.wait(timeout):
            raise TimeoutError('主线程执行超时')
        if 'e' in box:
            raise RuntimeError(box['e'])
        return box.get('r')

    @staticmethod
    def _settle(win, ms):
        """等待防抖重建完成（_recon_timer / FBP _timer）。"""
        if not ms:
            return
        from PySide6.QtWidgets import QApplication
        t0 = time.perf_counter()
        p = getattr(getattr(win, 'recon_widget', None), 'fbp_panel', None)
        while time.perf_counter() - t0 < ms / 1000.0:
            busy = False
            for t in (getattr(win, '_recon_timer', None),
                      getattr(p, '_timer', None) if p else None):
                try:
                    busy = busy or t.isActive()
                except Exception:
                    pass
            QApplication.processEvents()
            if not busy and time.perf_counter() - t0 > 0.3:
                break
            time.sleep(0.01)

    # ---------- op ----------
    def handle(self, req):
        op = str(req.get('op', ''))
        win = self.win
        if op == 'ping':
            return {'ok': True, 'pid': os.getpid(), 'port': self.port,
                    'app': 'simulate_ct6'}
        if op == 'list':
            return {'ok': True, 'variables': self.call_main(lambda: _inventory(win))}
        if op == 'get':
            def _g():
                return {'variables': _inventory(win), 'results': _results(win)}
            d = self.call_main(_g)
            d['ok'] = True
            return d
        if op == 'set':
            rep = self.call_main(lambda: self._do_set(req, win))
            return {'ok': True, 'applied': rep, 'results': self.call_main(lambda: _results(win))}
        if op in ('arch', 'scan'):
            key = req.get('arch') if op == 'arch' else None
            params = {'arch': key} if op == 'arch' else {'scan_mode': req.get('mode')}
            rep = self.call_main(lambda: self._do_set({'params': params,
                                                       'settle_ms': req.get('settle_ms', 0)}, win))
            return {'ok': True, 'applied': rep, 'results': self.call_main(lambda: _results(win))}
        if op == 'fbp':
            self.call_main(lambda: (_apply(win, {}), win.recon_widget.fbp_panel.run()))
            self.call_main(lambda: self._settle(win, req.get('settle_ms', 8000)))
            return {'ok': True, 'results': self.call_main(lambda: _results(win))}
        if op == 'fermi':
            fp = win.recon_widget.tab_fermi
            if fp is None:
                return {'ok': False, 'error': 'Fermi 页签不可用（仅光子计数架构可见）'}
            self.call_main(fp.compute)
            return {'ok': True}
        if op == 'tab':
            # 切换重建分析页签（Agent 需要"看到"某个页才能截图/核对）。
            # name 支持模糊匹配：sinogram / fbp / fermi / ideal / actual / artifact
            def _tab():
                tabs = win.recon_widget.tabs
                names = [tabs.tabText(i) for i in range(tabs.count())]
                key = str(req.get('name', req.get('tab', ''))).strip().lower()
                idx = next((i for i, t in enumerate(names) if key and key in t.lower()), None)
                if idx is None:
                    return {'ok': False, 'tabs': names}
                tabs.setCurrentIndex(idx)
                return {'ok': True, 'index': idx, 'name': names[idx], 'tabs': names}
            return self.call_main(_tab)
        if op == 'view':
            def _v():
                win.view.setCameraPosition(**({'distance': 2500})) if False else None
                return True
            return {'ok': True, 'note': 'view preset 需在 UI 手动或其 API 未暴露'}
        if op == 'shot':
            def _s():
                pm = win.grab()
                from PySide6.QtCore import QBuffer, QByteArray
                ba = QByteArray()
                buf = QBuffer(ba)
                buf.open(QBuffer.OpenModeFlag.WriteOnly)
                pm.save(buf, 'PNG')
                return base64.b64encode(bytes(ba)).decode('ascii')
            return {'ok': True, 'png_base64': self.call_main(_s)}
        return {'ok': False, 'error': f'未知 op: {op}'}

    def _do_set(self, req, win):
        rep = _apply(win, req.get('params'))
        # 触发重算（架构/扫描模式已在 setChecked/setCurrentText 里触发；这里兜底）
        try:
            win.update_simulation()
        except Exception as exc:
            rep['_update_simulation'] = f'error: {exc}'
        self._settle(win, req.get('settle_ms', 0))
        return rep

    # ---------- TCP ----------
    def _serve(self, conn):
        try:
            f = conn.makefile('rwb')
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    req = json.loads(line.decode('utf-8'))
                    resp = self.handle(req)
                except Exception as exc:
                    resp = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
                f.write((json.dumps(resp, ensure_ascii=False) + '\n').encode('utf-8'))
                f.flush()
        except Exception:
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def start(self):
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((self.host, self.port))
        self._srv.listen(4)
        self.port = self._srv.getsockname()[1]
        self._running = True

        def _loop():
            while self._running:
                try:
                    conn, _ = self._srv.accept()
                except OSError:
                    break
                th = threading.Thread(target=self._serve, args=(conn,), daemon=True)
                th.start()
                self._threads.append(th)

        threading.Thread(target=_loop, daemon=True).start()
        try:
            with open(DISCOVERY, 'w', encoding='utf-8') as fh:
                json.dump({'pid': os.getpid(), 'port': self.port, 'host': self.host,
                           'ts': time.time(), 'app': 'simulate_ct6'}, fh, indent=1)
        except Exception as exc:
            print(f'[WARN] 桥发现文件写入失败: {exc}')
        print(f'[bridge] simulate_ct6 控制桥已就绪: {self.host}:{self.port}  '
              f'(发现文件 {DISCOVERY})')
        return self

    def stop(self):
        self._running = False
        try:
            self._srv.close()
        except Exception:
            pass


_BRIDGE = None


def start_bridge(win, port=0):
    """在应用内启动控制桥（幂等）。"""
    global _BRIDGE
    if _BRIDGE is None:
        _BRIDGE = Bridge(win, port=port).start()
    return _BRIDGE


def client(port=None, host='127.0.0.1', timeout=200.0):
    """简易客户端：读发现文件或指定端口，发送 op 并取回响应。"""
    if port is None:
        with open(DISCOVERY, encoding='utf-8') as fh:
            d = json.load(fh)
        port, host = d['port'], d.get('host', host)

    def call(op, **kw):
        s = socket.create_connection((host, port), timeout=timeout)
        try:
            s.sendall((json.dumps({'op': op, **kw}, ensure_ascii=False) + '\n').encode('utf-8'))
            buf = b''
            while not buf.endswith(b'\n'):
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf += chunk
            return json.loads(buf.decode('utf-8'))
        finally:
            s.close()
    return call


if __name__ == '__main__':
    import sys
    c = client(int(sys.argv[1]) if len(sys.argv) > 1 else None)
    print(json.dumps(c(sys.argv[2] if len(sys.argv) > 2 else 'ping'), ensure_ascii=False,
                     indent=1)[:1200])
