    const payload = document.getElementById('trace-data').textContent;
    const TRACE = JSON.parse(payload);
    document.getElementById('payload').textContent = JSON.stringify(TRACE, null, 2);

    const tooltipEl = document.getElementById('tooltip');
    let _barRegistry = [];

    function fmt(us) {
      if (us === undefined || us === null) return '—';
      if (us >= 1000) return (us / 1000).toFixed(2) + ' ms';
      return us.toFixed(1) + ' µs';
    }

    function showCursorLine(frameId, xClient) {
      const frame = document.getElementById(frameId);
      if (!frame) return;
      let line = frame.querySelector('.cursor-line');
      if (!line) {
        line = document.createElement('div');
        line.className = 'cursor-line';
        frame.appendChild(line);
      }
      const frameRect = frame.getBoundingClientRect();
      line.style.display = 'block';
      line.style.left = (xClient - frameRect.left) + 'px';
    }

    function hideCursorLine(frameId) {
      const frame = document.getElementById(frameId);
      if (!frame) return;
      const line = frame.querySelector('.cursor-line');
      if (line) line.style.display = 'none';
    }

    function _installTooltip(canvas, registry) {
      if (canvas._tooltipInstalled) return;
      canvas._tooltipInstalled = true;
      const frame = canvas.closest('.chart-frame');
      const frameId = frame ? frame.id : '';
      canvas.addEventListener('mousemove', (event) => {
        const rect = canvas.getBoundingClientRect();
        const mx = event.clientX - rect.left;
        const my = event.clientY - rect.top;
        let best = null;
        for (const entry of registry) {
          if (mx >= entry.x && mx <= entry.x + entry.w && my >= entry.y && my <= entry.y + entry.h) {
            best = entry;
            break;
          }
        }
        if (best) {
          tooltipEl.style.display = 'block';
          tooltipEl.style.left = (event.clientX + 14) + 'px';
          tooltipEl.style.top = (event.clientY - 10) + 'px';
          tooltipEl.innerHTML = best.tip;
          showCursorLine(frameId, event.clientX);
        } else {
          tooltipEl.style.display = 'none';
          hideCursorLine(frameId);
        }
      });
      canvas.addEventListener('mouseleave', () => {
        tooltipEl.style.display = 'none';
        hideCursorLine(frameId);
      });
    }

    const chartState = new WeakMap();
    const palette = {
      compute: '#aed6f1',
      comm_collective: '#f9e79f',
      comm_p2p: '#fad7a0',
      data_move: '#a9dfbf',
      memory: '#d7bde2',
      unknown: '#d5d8dc',
      fwd: '#93c5fd',
      bwd: '#fca5a5',
      comm: '#fde68a',
      fsdp: '#c4b5fd',
      edge: '#94a3b8',
      explicit: '#dc2626',
    };
    const memoryPalette = {
      activation: '#60a5fa',
      allocation: '#c084fc',
      comm_buffer: '#f59e0b',
      comm_event_buffer: '#fbbf24',
      data_move: '#34d399',
      parameter: '#22c55e',
      gradient: '#fb7185',
      optimizer_state: '#a78bfa',
      unknown: '#94a3b8',
    };

    function formatBytes(bytes) {
      if (bytes === null || bytes === undefined || Number.isNaN(Number(bytes))) return 'n/a';
      let value = Number(bytes);
      const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
      for (const unit of units) {
        if (Math.abs(value) < 1024 || unit === 'TiB') {
          return unit === 'B' ? Math.round(value) + ' B' : value.toFixed(1) + ' ' + unit;
        }
        value /= 1024;
      }
      return value.toFixed(1) + ' TiB';
    }

    function shortName(name, maxLen = 42) {
      const cleaned = String(name || '').replace('aten.', '').replace('.default', '');
      return cleaned.length <= maxLen ? cleaned : cleaned.slice(0, maxLen - 1) + '…';
    }

    function eventStep(ev) {
      const metadata = ev.metadata || {};
      const value = metadata.step ?? ev.step ?? 0;
      const parsed = Number.parseInt(value, 10);
      return Number.isFinite(parsed) ? parsed : 0;
    }

    function eventLane(ev) {
      const eventType = String(ev.event_type || '');
      const metadata = ev.metadata || {};
      const strategy = String(metadata.strategy || '').toLowerCase();

      // DP gradient sync — separate lane per DP rank
      if (eventType.startsWith('dp_') || strategy === 'dp') {
        return 'DP rank ' + (ev.rank ?? 0);
      }
      // Optimizer step — separate lane per rank
      if (eventType.startsWith('optimizer') || eventType.includes('step')) {
        return 'Optim rank ' + (ev.rank ?? 0);
      }
      // TP all-reduce — separate lane
      if (eventType.startsWith('tp_') || strategy === 'tp') {
        return 'TP rank ' + (ev.rank ?? 0);
      }
      // FSDP events
      if (eventType.startsWith('fsdp_') || strategy === 'fsdp2' || strategy === 'fsdp') {
        return 'FSDP rank ' + (ev.rank ?? 0);
      }
      // PP events (and anything else with pp_stage / pp_rank)
      if (eventType.startsWith('pp_') || ev.pp_stage !== null && ev.pp_stage !== undefined) {
        const ppRank = ev.pp_rank ?? 0;
        const ppStage = ev.pp_stage;
        return 'PP stage ' + (ppStage ?? ppRank) + ' (rank ' + ppRank + ')';
      }
      // Loss compute on last stage
      if (eventType.startsWith('loss') || strategy === 'compute') {
        return 'Loss (pp rank ' + (ev.pp_rank ?? 0) + ')';
      }
      // Fallback
      return 'Rank ' + (ev.rank ?? 0);
    }

    function eventStrategy(ev) {
      const metadata = ev.metadata || {};
      const eventType = String(ev.event_type || '').toLowerCase();
      if (metadata.strategy) return String(metadata.strategy).toLowerCase();
      if (eventType.startsWith('fsdp_')) return 'fsdp';
      if (eventType.startsWith('tp_')) return 'tp';
      if (eventType.startsWith('dp_')) return 'dp';
      if (eventType.startsWith('pp_') || ev.pp_stage !== null && ev.pp_stage !== undefined) return 'pp';
      if (eventType.startsWith('loss')) return 'compute';
      if (eventType.startsWith('optimizer')) return 'optim';
      return 'other';
    }

    function scheduleRankViews(events) {
      const views = [{key: 'all', label: 'All ranks', kind: 'all'}];
      const seen = new Set(['all']);
      function add(view) {
        if (seen.has(view.key)) return;
        seen.add(view.key);
        views.push(view);
      }
      const ranks = Array.from(new Set(events.map((ev) => ev.rank).filter((rank) => rank !== null && rank !== undefined))).sort((a, b) => Number(a) - Number(b));
      for (const rank of ranks) add({key: 'global:' + rank, label: 'Global rank ' + rank, kind: 'global', rank});
      const ppStages = Array.from(new Set(events.map((ev) => ev.pp_stage).filter((rank) => rank !== null && rank !== undefined))).sort((a, b) => Number(a) - Number(b));
      for (const stage of ppStages) add({key: 'pp-stage:' + stage, label: 'PP stage ' + stage, kind: 'pp-stage', stage});
      const ppRanks = Array.from(new Set(events.map((ev) => ev.pp_rank).filter((rank) => rank !== null && rank !== undefined))).sort((a, b) => Number(a) - Number(b));
      for (const rank of ppRanks) add({key: 'pp-rank:' + rank, label: 'PP rank ' + rank, kind: 'pp-rank', ppRank: rank});
      for (const strategy of ['tp', 'dp', 'fsdp', 'fsdp2']) {
        const strategyRanks = Array.from(new Set(events.filter((ev) => eventStrategy(ev) === strategy).map((ev) => ev.rank).filter((rank) => rank !== null && rank !== undefined))).sort((a, b) => Number(a) - Number(b));
        const labelPrefix = strategy.toUpperCase();
        for (const rank of strategyRanks) add({key: 'strategy:' + strategy + ':' + rank, label: labelPrefix + ' rank ' + rank, kind: 'strategy', strategy, rank});
      }
      return views;
    }

    function rankViewMatches(ev, view) {
      if (!view || view.kind === 'all') return true;
      if (view.kind === 'global') return Number(ev.rank) === Number(view.rank);
      if (view.kind === 'pp-stage') return Number(ev.pp_stage) === Number(view.stage);
      if (view.kind === 'pp-rank') return Number(ev.pp_rank ?? ev.pp_stage) === Number(view.ppRank);
      if (view.kind === 'strategy') return eventStrategy(ev) === view.strategy && Number(ev.rank) === Number(view.rank);
      return true;
    }

    function scheduleEvents() {
      const events = [];
      for (const ev of TRACE.schedule?.events || []) events.push({...ev, name: ev.event_type});
      for (const ev of TRACE.fsdp_events || []) events.push({...ev, name: ev.event_type || 'fsdp'});
      for (const ev of TRACE.pp_events || []) events.push({...ev, name: ev.event_type || 'pp'});
      for (const ev of TRACE.comm_events || []) events.push({...ev, event_type: ev.op || 'comm', name: ev.op || 'comm'});
      return events.sort((a, b) => (Number(a.logical_clock || 0) - Number(b.logical_clock || 0)) || String(a.event_id || '').localeCompare(String(b.event_id || '')));
    }

    function renderRankTabs(canvas) {
      const step = Number.parseInt(canvas.dataset.step || '0', 10);
      const events = scheduleEvents().filter((ev) => eventStep(ev) === step);
      const views = scheduleRankViews(events);
      const tabs = document.querySelector('.rank-tabs[data-target="' + canvas.id + '"]');
      if (!tabs) return;
      const state = chartState.get(canvas) || {zoom: 1, rankView: 'all'};
      if (!views.some((view) => view.key === state.rankView)) state.rankView = 'all';
      chartState.set(canvas, state);
      tabs.textContent = '';
      for (const view of views) {
        const button = document.createElement('button');
        button.type = 'button';
        button.textContent = view.label;
        button.dataset.rankView = view.key;
        if (view.key === state.rankView) button.classList.add('active');
        button.addEventListener('click', () => {
          const current = chartState.get(canvas) || {zoom: 1};
          current.rankView = view.key;
          chartState.set(canvas, current);
          renderRankTabs(canvas);
          redraw(canvas);
        });
        tabs.appendChild(button);
      }
      const globalRanks = new Set(events.map((ev) => ev.rank).filter((rank) => rank !== null && rank !== undefined));
      if (globalRanks.size <= 1) {
        const onlyRank = Array.from(globalRanks)[0] ?? 0;
        const note = document.createElement('span');
        note.className = 'muted rank-note';
        note.textContent = 'Only local rank ' + onlyRank + ' is present in this trace; more rank tabs appear when multi-rank traces are captured or aggregated.';
        tabs.appendChild(note);
      }
    }

    function resizeCanvas(canvas, width, height) {
      const dpr = window.devicePixelRatio || 1;
      canvas.style.width = width + 'px';
      canvas.style.height = height + 'px';
      canvas.width = Math.ceil(width * dpr);
      canvas.height = Math.ceil(height * dpr);
      const ctx = canvas.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return ctx;
    }

    function roundedRect(ctx, x, y, w, h, r) {
      ctx.beginPath();
      ctx.moveTo(x + r, y);
      ctx.arcTo(x + w, y, x + w, y + h, r);
      ctx.arcTo(x + w, y + h, x, y + h, r);
      ctx.arcTo(x, y + h, x, y, r);
      ctx.arcTo(x, y, x + w, y, r);
      ctx.closePath();
    }

    function arrowLine(ctx, sx, sy, dx, dy, color, dashed = false, width = 1.2) {
      ctx.save();
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = width;
      ctx.globalAlpha = 0.8;
      if (dashed) ctx.setLineDash([5, 4]);
      ctx.beginPath();
      const mid = Math.max(30, Math.abs(dx - sx) / 2);
      ctx.moveTo(sx, sy);
      ctx.bezierCurveTo(sx + mid, sy, dx - mid, dy, dx, dy);
      ctx.stroke();
      const angle = Math.atan2(dy - sy, dx - sx);
      ctx.beginPath();
      ctx.moveTo(dx, dy);
      ctx.lineTo(dx - 8 * Math.cos(angle - Math.PI / 6), dy - 8 * Math.sin(angle - Math.PI / 6));
      ctx.lineTo(dx - 8 * Math.cos(angle + Math.PI / 6), dy - 8 * Math.sin(angle + Math.PI / 6));
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }

    function depColor(depType) {
      if (depType === 'pp_comm') return '#ea580c';
      if (depType === 'fsdp_comm') return '#7c3aed';
      if (depType === 'tp_comm') return '#0891b2';
      if (depType === 'dp_comm') return '#16a34a';
      if (depType === 'control') return '#475569';
      return palette.explicit;
    }

    function drawChromeTrace(canvas) {
      _barRegistry = [];
      const state = chartState.get(canvas) || {zoom: 1};
      chartState.set(canvas, state);
      const step = Number.parseInt(canvas.dataset.step || '0', 10);
      const allEvents = scheduleEvents().filter((ev) => eventStep(ev) === step);
      const hasTiming = allEvents.some(ev => ev.perf_cumulative_start_us !== undefined);

      // Build Chrome-trace-style event list: group by (pid, tid) = lane
      const laneMap = new Map();
      for (const ev of allEvents) {
        const lane = chromeTraceLane(ev);
        if (!laneMap.has(lane)) laneMap.set(lane, []);
        laneMap.get(lane).push(ev);
      }

      // Sort lanes
      const lanes = Array.from(laneMap.keys()).sort();
      // Sort events within each lane by cumulative start time
      for (const [lane, items] of laneMap) {
        items.sort((a, b) => {
          const ta = a.perf_cumulative_start_us !== undefined ? a.perf_cumulative_start_us : Number(a.logical_clock || 0);
          const tb = b.perf_cumulative_start_us !== undefined ? b.perf_cumulative_start_us : Number(b.logical_clock || 0);
          return ta - tb;
        });
      }

      // Compute time bounds
      const maxTime = hasTiming
        ? Math.max(1, ...allEvents.map(ev => (ev.perf_cumulative_start_us || 0) + (ev.perf_total_time_us || 0)))
        : Math.max(0, ...allEvents.map(ev => Number(ev.logical_clock || 0)));
      const pixelsPerUnit = hasTiming ? Math.max(0.005, 58 * state.zoom / Math.max(1, maxTime / 100)) : 58 * state.zoom;
      const laneH = 28;
      const padTop = 40;
      const labelW = 170;
      const desMemory = TRACE.metadata?.des_memory;
      const desStats = TRACE.metadata?.des_engine;
      const extraH = (desMemory ? laneH + 10 : 0) + (desStats ? 30 : 0);
      const width = Math.max(980, labelW + 80 + (maxTime + 10) * pixelsPerUnit);
      const height = Math.max(160, padTop + lanes.length * laneH + 24 + extraH);
      const ctx = resizeCanvas(canvas, width, height);
      ctx.clearRect(0, 0, width, height);
      ctx.font = '12px ui-sans-serif, system-ui, sans-serif';
      ctx.textBaseline = 'middle';

      // Note
      const note = document.querySelector('.chart-toolbar[data-target=\"' + canvas.id + '\"] .chart-note');
      const totalLabel = hasTiming ? (maxTime >= 1000 ? (maxTime / 1000).toFixed(2) + 'ms' : maxTime.toFixed(0) + 'µs') : maxTime + ' clocks';
      if (note) note.textContent = allEvents.length + ' events in ' + lanes.length + ' lanes. Total span: ' + totalLabel + '. Drag or use scrollbar to pan.';

      // Lane labels + backgrounds
      lanes.forEach((lane, idx) => {
        const y = padTop + idx * laneH;
        if (idx % 2 === 0) {
          ctx.fillStyle = '#f8fafc';
          ctx.fillRect(labelW, y - laneH / 2, width - labelW - 30, laneH);
        }
        // Show rank + PP annotation
        const rankNum = lane.replace(/^Rank /, '').split(' ')[0];
        const sample = laneMap.get(lane)?.[0];
        let annotation = lane;
        if (sample) {
          const ppR = sample.pp_rank;
          const ppS = sample.pp_stage;
          if (ppR !== null && ppR !== undefined) {
            annotation = 'Rank ' + rankNum + '  [PP' + ppR;
            if (ppS !== null && ppS !== undefined && ppS !== ppR) annotation += ' v' + ppS;
            annotation += ']';
          }
        }
        ctx.fillStyle = '#1e293b';
        ctx.font = 'bold 10px ui-sans-serif, system-ui, sans-serif';
        ctx.fillText(annotation, 6, y);
        ctx.strokeStyle = '#e2e8f0';
        ctx.beginPath();
        ctx.moveTo(labelW, y + laneH / 2);
        ctx.lineTo(width - 30, y + laneH / 2);
        ctx.stroke();
      });

      // Phase boundary shading
      const phaseRanges = {};
      for (const ev of allEvents) {
        const phase = _scheduleEventToPhase(ev.event_type, eventStrategy(ev));
        const tStart = ev.perf_cumulative_start_us || 0;
        const tEnd = tStart + (ev.perf_total_time_us || 0);
        if (!phaseRanges[phase]) phaseRanges[phase] = {min: tStart, max: tEnd};
        else {
          phaseRanges[phase].min = Math.min(phaseRanges[phase].min, tStart);
          phaseRanges[phase].max = Math.max(phaseRanges[phase].max, tEnd);
        }
      }
      const phaseColors = {forward: '#93c5fd', backward: '#fca5a5', optimizer: '#86efac'};
      for (const [phase, range] of Object.entries(phaseRanges)) {
        const x1 = labelW + range.min * pixelsPerUnit;
        const x2 = labelW + range.max * pixelsPerUnit;
        ctx.fillStyle = phaseColors[phase] || '#d5d8dc';
        ctx.globalAlpha = 0.15;
        ctx.fillRect(x1, padTop - 4, x2 - x1, lanes.length * laneH + 8);
        ctx.globalAlpha = 1.0;
      }

      // Time axis
      const axisY = padTop + lanes.length * laneH + 8;
      ctx.strokeStyle = '#94a3b8';
      ctx.beginPath();
      ctx.moveTo(labelW, axisY);
      ctx.lineTo(width - 30, axisY);
      ctx.stroke();
      const numTicks = Math.min(10, Math.ceil(maxTime / (hasTiming ? 500 : 5)));
      const tickInterval = Math.max(1, maxTime / numTicks);
      for (let t = 0; t <= maxTime; t += tickInterval) {
        const tx = labelW + t * pixelsPerUnit;
        ctx.fillStyle = '#64748b';
        ctx.font = '9px ui-sans-serif, system-ui, sans-serif';
        const tickLabel = hasTiming ? (t >= 1000 ? (t / 1000).toFixed(1) + 'ms' : Math.round(t) + 'µs') : Math.round(t);
        ctx.fillText(tickLabel, tx, axisY + 10);
        ctx.beginPath();
        ctx.moveTo(tx, axisY);
        ctx.lineTo(tx, axisY + 4);
        ctx.stroke();
      }

      // Draw bars
      for (const [lane, items] of laneMap) {
        const laneIdx = lanes.indexOf(lane);
        const y = padTop + laneIdx * laneH;
        for (const ev of items) {
          const tStart = hasTiming ? (ev.perf_cumulative_start_us || 0) : Number(ev.logical_clock || 0);
          const tDur = hasTiming ? Math.max(1, ev.perf_total_time_us || 0) : 1;
          const x = labelW + tStart * pixelsPerUnit;
          const barW = Math.max(4, tDur * pixelsPerUnit);
          const barH = laneH - 6;

          const name = String(ev.name || ev.event_type || '');
          let fill = palette.fwd;
          if (name.toLowerCase().includes('bwd')) fill = palette.bwd;
          else if (name.toLowerCase().includes('fsdp')) fill = palette.fsdp;
          else if (name.toLowerCase().includes('optim')) fill = '#86efac';
          else if (name.toLowerCase().includes('gradient')) fill = '#c084fc';
          else if (name.toLowerCase().includes('all_reduce')) fill = palette.comm;
          else if (name.toLowerCase().includes('send') || name.toLowerCase().includes('recv')) fill = '#f97316';

          ctx.fillStyle = fill;
          ctx.fillRect(x, y - barH / 2, barW, barH);
          ctx.strokeStyle = '#334155';
          ctx.lineWidth = 0.5;
          ctx.strokeRect(x, y - barH / 2, barW, barH);
          _barRegistry.push({
            x: x, y: y - barH / 2, w: barW, h: barH,
            tip: '<b>' + shortName(ev.op_name || ev.event_type || name) + '</b><br>' +
                 'Phase: ' + (ev.phase || '—') + '<br>' +
                 'Start: ' + fmt(ev.perf_cumulative_start_us) + '<br>' +
                 'Duration: ' + fmt(ev.perf_total_time_us) + '<br>' +
                 'Engine: ' + eventEngineType(ev.event_type) +
                 (ev.op_node_ids && ev.op_node_ids.length ? '<br>Nodes: ' + ev.op_node_ids.length : '')
          });

          // Name inside bar (if wide enough) or beside
          if (barW > 50) {
            ctx.fillStyle = '#1e293b';
            ctx.font = '9px ui-sans-serif, system-ui, sans-serif';
            ctx.fillText(shortName(name, 18), x + 3, y);
          }
        }
      }

      // Memory track
      if (desMemory && desMemory.timeline && desMemory.timeline.length > 0) {
        const memLaneH = 20;
        const memLaneIdx = lanes.length;
        const memLaneY = padTop + memLaneIdx * laneH + 6;
        const peakBytes = desMemory.peak_total_bytes || 1;
        ctx.fillStyle = '#1e293b';
        ctx.font = 'bold 10px ui-sans-serif, system-ui, sans-serif';
        ctx.fillText('Memory', 6, memLaneY);
        ctx.strokeStyle = '#e2e8f0';
        ctx.beginPath();
        ctx.moveTo(labelW, memLaneY + laneH / 2);
        ctx.lineTo(width - 30, memLaneY + laneH / 2);
        ctx.stroke();
        const barH = memLaneH - 4;
        const staticRatio = (desMemory.static_memory_bytes || 0) / peakBytes;
        ctx.fillStyle = '#64748b';
        ctx.globalAlpha = 0.5;
        const staticH = barH * staticRatio;
        ctx.fillRect(labelW, memLaneY - staticH / 2 + barH / 2 - staticH / 2, width - labelW - 30, staticH);
        ctx.globalAlpha = 1.0;
        for (const sample of desMemory.timeline) {
          const x = labelW + sample.time_us * pixelsPerUnit;
          const w = Math.max(2, (sample.duration_us || 0) * pixelsPerUnit);
          const dynRatio = sample.dynamic_bytes / peakBytes;
          ctx.fillStyle = '#60a5fa';
          ctx.globalAlpha = 0.3;
          ctx.fillRect(x, memLaneY - barH / 2, w, barH * dynRatio);
          ctx.globalAlpha = 1.0;
        }
      }

      // DES stats bar
      if (desStats) {
        const statsY = height - 20;
        ctx.fillStyle = '#1e293b';
        ctx.font = '10px ui-sans-serif, system-ui, sans-serif';
        ctx.fillText('DES:', 6, statsY);
        ctx.fillStyle = '#4fc3f7';
        ctx.fillText('Compute ' + (desStats.compute_busy_pct || 0).toFixed(1) + '%', 46, statsY);
        ctx.fillStyle = '#f9e79f';
        ctx.fillText('Comm ' + (desStats.comm_busy_pct || 0).toFixed(1) + '%', 140, statsY);
        ctx.fillStyle = '#dc2626';
        ctx.fillText('Overlap ' + (desStats.overlap_pct || 0).toFixed(1) + '%', 220, statsY);
        ctx.fillStyle = '#e5e7eb';
        const e2e = desStats.e2e_step_time_us || 0;
        ctx.fillText('E2E ' + (e2e >= 1000 ? (e2e / 1000).toFixed(2) + 'ms' : e2e.toFixed(0) + 'µs'), 320, statsY);
      }
      _installTooltip(canvas, _barRegistry);
    }

    function eventEngineType(eventType) {
      const et = String(eventType || '').toLowerCase();
      if (et.startsWith('pp_send') || et.startsWith('pp_recv') ||
          et.startsWith('fsdp2_all_gather') || et.startsWith('fsdp2_reduce_scatter') ||
          et.startsWith('dp_gradient') || et.startsWith('tp_')) {
        return 'comm';
      }
      return 'compute';
    }

    function _scheduleEventToPhase(eventType, strategy) {
      const et = (eventType || '').toLowerCase();
      const st = (strategy || '').toLowerCase();
      if (et.includes('bwd') || et.includes('backward') || st.includes('backward')) return 'backward';
      if (et.includes('fwd') || et.includes('forward') || st === 'pp' || st === 'compute') return 'forward';
      if (et.includes('optim')) return 'optimizer';
      if (et.includes('reduce') || et.includes('gradient')) return 'backward';
      return 'forward';
    }

    function chromeTraceLane(ev) {
      const eventType = String(ev.event_type || '');
      const rank = ev.rank ?? 0;
      const engine = eventEngineType(eventType);
      return 'Rank ' + rank + ' ' + (engine === 'comm' ? 'Comm' : 'Compute');
    }

    function drawDag(canvas) {
      const state = chartState.get(canvas) || {zoom: 1};
      chartState.set(canvas, state);
      const phase = canvas.dataset.phase || 'unknown';
      const maxNodes = Number.parseInt(canvas.dataset.maxNodes || '220', 10);
      const allNodes = TRACE.compute_graph?.nodes || [];
      const phaseNodes = allNodes.filter((node) => (node.phase || 'unknown') === phase).slice(0, maxNodes);
      const nodeIds = new Set(phaseNodes.map((node) => node.node_id));
      const edges = (TRACE.compute_graph?.edges || []).filter((edge) => nodeIds.has(edge.src) && nodeIds.has(edge.dst));

      const hasDes = phaseNodes.some((node) => node.des_start_time_us !== null && node.des_start_time_us !== undefined);

      if (hasDes) {
        drawDagDesTemporal(canvas, phaseNodes, edges, state, phase);
      } else {
        drawDagTopological(canvas, phaseNodes, edges, state, phase);
      }
    }

    function drawDagDesTemporal(canvas, phaseNodes, edges, state, phase) {
      _barRegistry = [];
      const computeNodes = phaseNodes.filter(n => !['comm_collective', 'comm_p2p'].includes(n.op_type));
      const commNodes = phaseNodes.filter(n => ['comm_collective', 'comm_p2p'].includes(n.op_type));

      const maxTime = Math.max(...phaseNodes.map(n => n.des_finish_time_us || 0));
      const minTime = Math.min(...phaseNodes.map(n => n.des_start_time_us || 0));
      const timeRange = Math.max(1, maxTime - minTime);

      const pixelsPerUnit = 58 * state.zoom / Math.max(1, timeRange / 100);
      const labelW = 140;
      const computeRowY = 50;
      const commRowY = 110;
      const nodeH = 36;
      const width = Math.max(1100, labelW + 40 + timeRange * pixelsPerUnit + 100);
      const memoryMarkerY = commRowY + 50;
      const height = Math.max(200, memoryMarkerY + 30);
      const ctx = resizeCanvas(canvas, width, height);
      ctx.clearRect(0, 0, width, height);

      ctx.strokeStyle = '#94a3b8';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(labelW, 20);
      ctx.lineTo(width - 30, 20);
      ctx.stroke();
      const numTicks = Math.min(10, Math.ceil(timeRange / 50));
      const tickInterval = Math.max(1, timeRange / numTicks);
      for (let t = minTime; t <= maxTime; t += tickInterval) {
        const x = labelW + (t - minTime) * pixelsPerUnit;
        ctx.fillStyle = '#64748b';
        ctx.font = '9px ui-sans-serif, system-ui, sans-serif';
        const label = t >= 1000 ? (t / 1000).toFixed(1) + 'ms' : Math.round(t) + 'µs';
        ctx.fillText(label, x, 12);
        ctx.beginPath();
        ctx.moveTo(x, 18);
        ctx.lineTo(x, 22);
        ctx.stroke();
      }

      ctx.fillStyle = '#1e293b';
      ctx.font = 'bold 10px ui-sans-serif, system-ui, sans-serif';
      ctx.fillText('Compute Engine', 6, computeRowY);
      ctx.fillText('Comm Engine', 6, commRowY);

      const positions = new Map();
      for (const node of computeNodes) {
        const start = node.des_start_time_us || minTime;
        const finish = node.des_finish_time_us || start;
        const x = labelW + (start - minTime) * pixelsPerUnit;
        const w = Math.max(4, (finish - start) * pixelsPerUnit);
        positions.set(node.node_id, {x, y: computeRowY - nodeH / 2, w, endX: labelW + (finish - minTime) * pixelsPerUnit});
        const fill = palette[node.op_type] || palette.unknown;
        const isContended = finish - start > (node.perf_result?.total_time_us || 0) + 0.1;
        roundedRect(ctx, x, computeRowY - nodeH / 2, w, nodeH, 4);
        ctx.fillStyle = fill;
        ctx.fill();
        ctx.strokeStyle = isContended ? '#dc2626' : '#334155';
        ctx.lineWidth = isContended ? 2 : 0.5;
        ctx.stroke();
        _barRegistry.push({
          x: x, y: computeRowY - nodeH / 2, w: w, h: nodeH,
          tip: '<b>' + shortName(node.op_name) + '</b><br>' +
               'Type: ' + node.op_type + '<br>' +
               'Phase: ' + node.phase + '<br>' +
               (node.des_start_time_us ? 'DES start: ' + fmt(node.des_start_time_us) + '<br>' : '') +
               (node.des_finish_time_us ? 'DES finish: ' + fmt(node.des_finish_time_us) + '<br>' : '') +
               (node.perf_result ? 'Duration: ' + fmt(node.perf_result.total_time_us) + '<br>' : '')
        });
        ctx.fillStyle = '#0f172a';
        ctx.font = '700 9px ui-sans-serif, system-ui, sans-serif';
        if (w > 30) ctx.fillText(shortName(node.op_name, 12), x + 3, computeRowY - 6);
        if (w > 60) {
          ctx.font = 'italic 8px ui-sans-serif, system-ui, sans-serif';
          const dur = finish - start;
          ctx.fillText(dur >= 1000 ? (dur / 1000).toFixed(1) + 'ms' : dur.toFixed(0) + 'µs', x + 3, computeRowY + 8);
        }
      }

      for (const node of commNodes) {
        const start = node.des_start_time_us || minTime;
        const finish = node.des_finish_time_us || start;
        const x = labelW + (start - minTime) * pixelsPerUnit;
        const w = Math.max(4, (finish - start) * pixelsPerUnit);
        positions.set(node.node_id, {x, y: commRowY - nodeH / 2, w, endX: labelW + (finish - minTime) * pixelsPerUnit});
        const fill = palette[node.op_type] || palette.unknown;
        const isContended = finish - start > (node.perf_result?.total_time_us || 0) + 0.1;
        roundedRect(ctx, x, commRowY - nodeH / 2, w, nodeH, 4);
        ctx.fillStyle = fill;
        ctx.fill();
        ctx.strokeStyle = isContended ? '#dc2626' : '#334155';
        ctx.lineWidth = isContended ? 2 : 0.5;
        ctx.stroke();
        _barRegistry.push({
          x: x, y: commRowY - nodeH / 2, w: w, h: nodeH,
          tip: '<b>' + shortName(node.op_name) + '</b><br>' +
               'Type: ' + node.op_type + '<br>' +
               'Phase: ' + node.phase + '<br>' +
               (node.des_start_time_us ? 'DES start: ' + fmt(node.des_start_time_us) + '<br>' : '') +
               (node.des_finish_time_us ? 'DES finish: ' + fmt(node.des_finish_time_us) + '<br>' : '') +
               (node.perf_result ? 'Duration: ' + fmt(node.perf_result.total_time_us) + '<br>' : '')
        });
        ctx.fillStyle = '#0f172a';
        ctx.font = '700 9px ui-sans-serif, system-ui, sans-serif';
        if (w > 30) ctx.fillText(shortName(node.op_name, 12), x + 3, commRowY - 6);
        if (w > 60) {
          ctx.font = 'italic 8px ui-sans-serif, system-ui, sans-serif';
          const dur = finish - start;
          ctx.fillText(dur >= 1000 ? (dur / 1000).toFixed(1) + 'ms' : dur.toFixed(0) + 'µs', x + 3, commRowY + 8);
        }
      }

      for (const edge of edges) {
        const src = positions.get(edge.src);
        const dst = positions.get(edge.dst);
        if (!src || !dst) continue;
        arrowLine(ctx, src.endX, src.y + nodeH / 2, dst.x - 4, dst.y + nodeH / 2, edge.type === 'data' ? '#2563eb' : '#64748b', edge.type !== 'data', 1);
      }

      const desMemory = TRACE.metadata?.des_memory;
      if (desMemory && desMemory.timeline) {
        const memoryPaletteMap = {activation: '#60a5fa', gradient: '#fb7185', comm_buffer: '#f59e0b', optimizer_state: '#a78bfa', parameter: '#22c55e'};
        const peakMem = desMemory.peak_total_bytes || 1;
        const memBarH = 12;
        for (const sample of desMemory.timeline) {
          const xStart = labelW + (sample.time_us - minTime) * pixelsPerUnit;
          const w = Math.max(2, (sample.duration_us || 0) * pixelsPerUnit);
          if (xStart + w < labelW || xStart > width - 30) continue;
          const dynRatio = (sample.dynamic_bytes || 0) / peakMem;
          ctx.fillStyle = '#60a5fa';
          ctx.globalAlpha = 0.25;
          ctx.fillRect(xStart, memoryMarkerY - memBarH / 2, w, memBarH * dynRatio);
          ctx.globalAlpha = 1.0;
        }
      }
      _installTooltip(canvas, _barRegistry);
    }

    function drawDagTopological(canvas, phaseNodes, edges, state, phase) {
      _barRegistry = [];
      const preds = new Map(phaseNodes.map((node) => [node.node_id, []]));
      const succs = new Map(phaseNodes.map((node) => [node.node_id, []]));
      const indeg = new Map(phaseNodes.map((node) => [node.node_id, 0]));
      for (const edge of edges) {
        preds.get(edge.dst)?.push(edge.src);
        succs.get(edge.src)?.push(edge.dst);
        indeg.set(edge.dst, (indeg.get(edge.dst) || 0) + 1);
      }
      const queue = phaseNodes.filter((node) => (indeg.get(node.node_id) || 0) === 0).map((node) => node.node_id);
      const depth = new Map(phaseNodes.map((node) => [node.node_id, 0]));
      for (let i = 0; i < queue.length; i++) {
        const id = queue[i];
        for (const dst of succs.get(id) || []) {
          depth.set(dst, Math.max(depth.get(dst) || 0, (depth.get(id) || 0) + 1));
          indeg.set(dst, (indeg.get(dst) || 0) - 1);
          if (indeg.get(dst) === 0) queue.push(dst);
        }
      }
      phaseNodes.forEach((node, idx) => {
        if (!queue.includes(node.node_id)) depth.set(node.node_id, Math.floor(idx / 8));
      });

      const byDepth = new Map();
      for (const node of phaseNodes) {
        const d = depth.get(node.node_id) || 0;
        if (!byDepth.has(d)) byDepth.set(d, []);
        byDepth.get(d).push(node);
      }
      const maxDepth = Math.max(0, ...Array.from(byDepth.keys()));
      const maxLayer = Math.max(1, ...Array.from(byDepth.values()).map((items) => items.length));
      const colW = 250 * state.zoom;
      const rowH = 78;
      const width = Math.max(1100, 80 + (maxDepth + 1) * colW);
      const height = Math.max(180, 80 + maxLayer * rowH);
      const ctx = resizeCanvas(canvas, width, height);
      ctx.clearRect(0, 0, width, height);

      const positions = new Map();
      for (const [d, items] of byDepth.entries()) {
        items.forEach((node, idx) => {
          positions.set(node.node_id, {x: 32 + d * colW, y: 36 + idx * rowH});
        });
      }

      for (const edge of edges) {
        const src = positions.get(edge.src);
        const dst = positions.get(edge.dst);
        if (!src || !dst) continue;
        const srcNode = phaseNodes.find(n => n.node_id === edge.src);
        const dstNode = phaseNodes.find(n => n.node_id === edge.dst);
        const srcDur = Number(((srcNode || {}).perf_result || {}).total_time_us || 0);
        const maxDur = Math.max(1, ...phaseNodes.map(n => Number((n.perf_result || {}).total_time_us || 0)));
        const srcW = 140 + (srcDur > 0 ? Math.max(0.15, Math.log2(1 + srcDur) / Math.log2(1 + maxDur)) * 120 : 0);
        arrowLine(ctx, src.x + srcW, src.y + 28, dst.x - 6, dst.y + 28, edge.type === 'data' ? '#2563eb' : '#64748b', edge.type === 'control', 1);
      }

      for (const node of phaseNodes) {
        const pos = positions.get(node.node_id);
        const fill = palette[node.op_type] || palette.unknown;
        const pr = node.perf_result || {};
        const durUs = Number(pr.total_time_us || 0);
        const maxDur = Math.max(1, ...phaseNodes.map(n => Number((n.perf_result || {}).total_time_us || 0)));
        const logScale = durUs > 0 ? Math.max(0.15, Math.log2(1 + durUs) / Math.log2(1 + maxDur)) : 0.15;
        const nodeW = 140 + logScale * 120;
        roundedRect(ctx, pos.x, pos.y, nodeW, 56, 8);
        ctx.fillStyle = fill;
        ctx.fill();
        ctx.strokeStyle = '#334155';
        ctx.lineWidth = 1;
        ctx.stroke();
        _barRegistry.push({
          x: pos.x, y: pos.y, w: nodeW, h: 56,
          tip: '<b>' + shortName(node.op_name) + '</b><br>' +
               'Type: ' + (node.op_type || 'unknown') + '<br>' +
               'Phase: ' + (node.phase || 'unknown') + '<br>' +
               (durUs > 0 ? 'Duration: ' + fmt(durUs) + '<br>' : '')
        });
        ctx.fillStyle = '#0f172a';
        ctx.font = '700 12px ui-sans-serif, system-ui, sans-serif';
        ctx.fillText(shortName(node.op_name, 28), pos.x + 8, pos.y + 17);
        const shape = (node.outputs || []).slice(0, 1).map((t) => '[' + (t.shape || []).join(',') + ']').join(', ');
        ctx.fillStyle = '#334155';
        ctx.font = '10px ui-sans-serif, system-ui, sans-serif';
        ctx.fillText(shortName((node.op_type || 'unknown') + ' ' + shape, 32), pos.x + 8, pos.y + 32);
        if (durUs > 0) {
          const timeLabel = durUs >= 1000 ? (durUs / 1000).toFixed(2) + 'ms' : durUs.toFixed(1) + 'µs';
          ctx.fillStyle = durUs > 50 ? '#b91c1c' : '#047857';
          ctx.font = 'italic 10px ui-sans-serif, system-ui, sans-serif';
          ctx.fillText(timeLabel, pos.x + 8, pos.y + 47);
        }
      }
      _installTooltip(canvas, _barRegistry);
    }

    function memoryEvents() {
      return TRACE.memory_events || [];
    }

    function memoryCategoryColor(category) {
      return memoryPalette[category] || memoryPalette.unknown;
    }

    function memoryLifetimeLabel(event) {
      const start = event.lifetime_start;
      const end = event.lifetime_end;
      if (start === null || start === undefined || end === null || end === undefined) return 'resident';
      return String(start) + ' → ' + String(end);
    }

    function memoryCategories(events) {
      const preferred = [
        'parameter',
        'optimizer_state',
        'gradient',
        'activation',
        'allocation',
        'data_move',
        'comm_buffer',
        'comm_event_buffer',
      ];
      const present = new Set(events.map((event) => event.category || 'unknown'));
      const ordered = preferred.filter((category) => present.has(category));
      for (const category of Array.from(present).sort()) {
        if (!ordered.includes(category)) ordered.push(category);
      }
      return ordered;
    }

    function buildMemorySamples(events) {
      const lifetimed = events.filter((event) =>
        event.lifetime_start !== null && event.lifetime_start !== undefined &&
        event.lifetime_end !== null && event.lifetime_end !== undefined
      );
      const resident = events.filter((event) =>
        event.lifetime_start === null || event.lifetime_start === undefined ||
        event.lifetime_end === null || event.lifetime_end === undefined
      );
      const categories = memoryCategories(events);
      const maxIndex = Math.max(0, ...lifetimed.map((event) => Number(event.lifetime_end || 0)));
      const residentByCategory = new Map(categories.map((category) => [category, 0]));
      for (const event of resident) {
        const category = event.category || 'unknown';
        residentByCategory.set(category, (residentByCategory.get(category) || 0) + Number(event.bytes || 0));
      }

      const samples = [];
      let peak = 0;
      for (let idx = 0; idx <= maxIndex; idx++) {
        const byCategory = new Map(residentByCategory);
        for (const event of lifetimed) {
          const start = Number(event.lifetime_start);
          const end = Number(event.lifetime_end);
          if (idx < start || idx > end) continue;
          const category = event.category || 'unknown';
          byCategory.set(category, (byCategory.get(category) || 0) + Number(event.bytes || 0));
        }
        const total = Array.from(byCategory.values()).reduce((acc, value) => acc + value, 0);
        peak = Math.max(peak, total);
        samples.push({idx, byCategory, total});
      }
      return {samples, categories, peak, residentTotal: Array.from(residentByCategory.values()).reduce((acc, value) => acc + value, 0)};
    }

    function buildDesMemorySamples(desMemory, events) {
      const categories = memoryCategories(events);
      const timeline = desMemory.timeline || [];
      const staticBytes = desMemory.static_memory_bytes || 0;
      const peak = desMemory.peak_total_bytes || 0;
      const residentTotal = staticBytes;
      const samples = timeline.map(s => ({time_us: s.time_us, static_bytes: s.static_bytes, dynamic_bytes: s.dynamic_bytes, total_bytes: s.total_bytes, byCategory: new Map(Object.entries(s.by_category || {})), total: s.total_bytes}));
      return {samples, categories, peak, residentTotal, hasDes: true, staticBytes};
    }

    function drawMemoryTrace(canvas) {
      _barRegistry = [];
      const state = chartState.get(canvas) || {zoom: 1};
      chartState.set(canvas, state);
      const events = memoryEvents();
      const desMemory = TRACE.metadata?.des_memory;
      const useDes = desMemory && desMemory.timeline && desMemory.timeline.length > 0;
      const data = useDes ? buildDesMemorySamples(desMemory, events) : buildMemorySamples(events);
      const {samples, categories, peak, residentTotal} = data;
      const hasDes = data.hasDes || false;
      const staticBytes = data.staticBytes || 0;
      const scale = 14 * state.zoom;
      const plotLeft = 90;
      const plotTop = 38;
      const plotHeight = 250;
      const legendTop = plotTop + plotHeight + 42;
      let width, minTime, timeRange, pixelsPerTime, plotWidth;
      if (hasDes) {
        minTime = Math.min(...samples.map(s => s.time_us));
        timeRange = Math.max(1, Math.max(...samples.map(s => s.time_us)) - minTime);
        pixelsPerTime = 58 * state.zoom / Math.max(1, timeRange / 100);
        width = Math.max(980, plotLeft + 90 + timeRange * pixelsPerTime);
        plotWidth = width - plotLeft - 90;
      } else {
        width = Math.max(980, plotLeft + 90 + Math.max(1, samples.length) * scale);
        plotWidth = width - plotLeft - 90;
      }
      const height = 390;
      const ctx = resizeCanvas(canvas, width, height);
      ctx.clearRect(0, 0, width, height);
      ctx.font = '12px ui-sans-serif, system-ui, sans-serif';
      ctx.textBaseline = 'middle';

      const note = document.querySelector('.chart-toolbar[data-target="' + canvas.id + '"] .chart-note');
      if (note) {
        if (hasDes) {
          note.textContent = samples.length + ' DES memory samples. Peak: ' + formatBytes(peak) + ' (static ' + formatBytes(staticBytes) + ', dynamic peak ' + formatBytes(peak - staticBytes) + ').';
        } else {
          const lifetimedCount = events.filter((event) =>
            event.lifetime_start !== null && event.lifetime_start !== undefined &&
            event.lifetime_end !== null && event.lifetime_end !== undefined
          ).length;
          note.textContent = lifetimedCount + ' lifetimed events, ' + events.length +
            ' total. Estimated total peak including resident baseline: ' + formatBytes(peak) +
            ' (resident baseline ' + formatBytes(residentTotal) + ').';
        }
      }

      ctx.fillStyle = '#0f172a';
      ctx.font = '700 13px ui-sans-serif, system-ui, sans-serif';
      if (hasDes) {
        ctx.fillText('Estimated live memory by DES wall-clock time', plotLeft, 18);
        ctx.font = '11px ui-sans-serif, system-ui, sans-serif';
        ctx.fillStyle = '#475569';
        ctx.fillText('x: DES time, y: bytes. Static memory shown as baseline band. Phase boundaries marked with dashed lines.', plotLeft, 34);
      } else {
        ctx.fillText('Estimated live memory by operator order', plotLeft, 18);
        ctx.font = '11px ui-sans-serif, system-ui, sans-serif';
        ctx.fillStyle = '#475569';
        ctx.fillText('x: graph node order, y: bytes. Resident model-state estimates are drawn as a baseline.', plotLeft, 34);
      }

      ctx.strokeStyle = '#94a3b8';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(plotLeft, plotTop);
      ctx.lineTo(plotLeft, plotTop + plotHeight);
      ctx.lineTo(width - 36, plotTop + plotHeight);
      ctx.stroke();

      const safePeak = Math.max(1, peak);
      for (let tick = 0; tick <= 4; tick++) {
        const value = safePeak * tick / 4;
        const y = plotTop + plotHeight - (value / safePeak) * plotHeight;
        ctx.strokeStyle = tick === 0 ? '#94a3b8' : '#e2e8f0';
        ctx.beginPath();
        ctx.moveTo(plotLeft, y);
        ctx.lineTo(width - 36, y);
        ctx.stroke();
        ctx.fillStyle = '#334155';
        ctx.fillText(formatBytes(value), 8, y);
      }

      if (hasDes && staticBytes > 0) {
        const staticH = (staticBytes / safePeak) * plotHeight;
        const staticY = plotTop + plotHeight - staticH;
        ctx.fillStyle = '#64748b';
        ctx.globalAlpha = 0.35;
        ctx.fillRect(plotLeft, staticY, plotWidth, staticH);
        ctx.globalAlpha = 1.0;
        ctx.fillStyle = '#334155';
        ctx.font = '9px ui-sans-serif, system-ui, sans-serif';
        ctx.fillText('static ' + formatBytes(staticBytes), plotLeft + 3, staticY + 10);
      }

      if (hasDes) {
        const phaseBoundaries = new Map();
        for (const node of TRACE.compute_graph?.nodes || []) {
          const phase = node.phase || 'unknown';
          const start = node.des_start_time_us;
          if (start !== null && start !== undefined) {
            if (!phaseBoundaries.has(phase) || start < phaseBoundaries.get(phase)) {
              phaseBoundaries.set(phase, start);
            }
          }
        }
        const phaseColors = {forward: '#93c5fd', backward: '#fca5a5', optimizer: '#86efac'};
        for (const [phase, time] of phaseBoundaries) {
          const x = plotLeft + (time - minTime) * pixelsPerTime;
          if (x < plotLeft || x > plotLeft + plotWidth) continue;
          ctx.strokeStyle = phaseColors[phase] || '#94a3b8';
          ctx.lineWidth = 1;
          ctx.setLineDash([5, 4]);
          ctx.beginPath();
          ctx.moveTo(x, plotTop);
          ctx.lineTo(x, plotTop + plotHeight);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.fillStyle = '#475569';
          ctx.font = '9px ui-sans-serif, system-ui, sans-serif';
          ctx.fillText(phase, x + 3, plotTop - 6);
        }
      }

      for (const sample of samples) {
        let yTop = plotTop + plotHeight;
        let x, barW;
        if (hasDes) {
          x = plotLeft + (sample.time_us - minTime) * pixelsPerTime;
          barW = Math.max(2, (sample.duration_us || 0) * pixelsPerTime);
        } else {
          x = plotLeft + sample.idx * scale;
          barW = Math.max(2, scale - 1);
        }
        for (const category of categories) {
          const bytes = sample.byCategory.get(category) || 0;
          if (bytes <= 0) continue;
          const h = bytes / safePeak * plotHeight;
          yTop -= h;
          ctx.fillStyle = memoryCategoryColor(category);
          ctx.fillRect(x, yTop, barW, h);
        }
        _barRegistry.push({
          x: x, y: yTop, w: barW, h: plotTop + plotHeight - yTop,
          tip: '<b>Memory @ ' + fmt(sample.time_us) + '</b><br>' +
               'Dynamic: ' + formatBytes(sample.dynamic_bytes || 0) + '<br>' +
               'Total: ' + formatBytes(sample.total_bytes || sample.total || sample.dynamic_bytes || 0) + '<br>' +
               (sample.duration_us ? 'Span: ' + fmt(sample.duration_us) + '<br>' : '') +
               (sample.byCategory ? Object.entries(sample.byCategory instanceof Map ? Array.from(sample.byCategory.entries()) : sample.byCategory).filter(([k,b]) => b > 0).map(([cat,b]) => cat + ': ' + formatBytes(b)).join('<br>') : '')
        });
      }

      ctx.fillStyle = '#334155';
      ctx.font = '11px ui-sans-serif, system-ui, sans-serif';
      if (hasDes) {
        const maxTimeVal = minTime + timeRange;
        const numTicks = Math.min(8, Math.ceil(timeRange / 50));
        const tickInterval = Math.max(1, timeRange / numTicks);
        for (let t = minTime; t <= maxTimeVal; t += tickInterval) {
          const x = plotLeft + (t - minTime) * pixelsPerTime;
          const label = t >= 1000 ? (t / 1000).toFixed(1) + 'ms' : Math.round(t) + 'µs';
          ctx.fillText(label, x, plotTop + plotHeight + 18);
        }
      } else {
        const maxIdx = Math.max(0, samples.length - 1);
        for (const idx of [0, Math.floor(maxIdx / 2), maxIdx]) {
          const x = plotLeft + idx * scale;
          ctx.fillText(String(idx), x, plotTop + plotHeight + 18);
        }
      }

      let legendX = plotLeft;
      let legendY = legendTop;
      for (const category of categories) {
        if (legendX > width - 220) {
          legendX = plotLeft;
          legendY += 22;
        }
        ctx.fillStyle = memoryCategoryColor(category);
        ctx.fillRect(legendX, legendY - 6, 12, 12);
        ctx.fillStyle = '#0f172a';
        ctx.fillText(category, legendX + 18, legendY);
        legendX += 150;
      }
      _installTooltip(canvas, _barRegistry);
    }

    function populateMemoryTable() {
      const body = document.getElementById('memory-events-body');
      if (!body) return;
      body.textContent = '';
      const events = memoryEvents()
        .slice()
        .sort((a, b) => Number(b.bytes || 0) - Number(a.bytes || 0))
        .slice(0, 120);
      for (const event of events) {
        const row = document.createElement('tr');
        const values = [
          event.event_id || '',
          event.category || 'unknown',
          event.phase || 'unknown',
          formatBytes(event.bytes || 0),
          memoryLifetimeLabel(event),
          event.node_id || '',
        ];
        for (const value of values) {
          const cell = document.createElement('td');
          cell.textContent = value;
          row.appendChild(cell);
        }
        body.appendChild(row);
      }
    }

    function redraw(canvas) {
      if (canvas.classList.contains('chrome-trace-chart')) drawChromeTrace(canvas);
      else if (canvas.classList.contains('memory-chart')) drawMemoryTrace(canvas);
      else drawDag(canvas);
    }

    function installChart(canvas) {
      chartState.set(canvas, {zoom: 1, rankView: 'all'});
      const frame = canvas.closest('.chart-frame');
      let dragging = false;
      let startX = 0;
      let startScroll = 0;
      frame.addEventListener('mousedown', (event) => {
        dragging = true;
        startX = event.clientX;
        startScroll = frame.scrollLeft;
        frame.classList.add('dragging');
      });
      window.addEventListener('mouseup', () => {
        dragging = false;
        frame.classList.remove('dragging');
      });
      window.addEventListener('mousemove', (event) => {
        if (!dragging) return;
        frame.scrollLeft = startScroll - (event.clientX - startX);
      });
      frame.addEventListener('wheel', (event) => {
        if (!event.ctrlKey && Math.abs(event.deltaX) < Math.abs(event.deltaY)) return;
        event.preventDefault();
        frame.scrollLeft += event.deltaX || event.deltaY;
      }, {passive: false});
      if (canvas.classList.contains('chrome-trace-chart')) {
        // No rank tabs needed for Chrome trace view
      }
      redraw(canvas);
    }

    document.querySelectorAll('canvas.trace-chart').forEach(installChart);
    populateMemoryTable();
    document.querySelectorAll('.chart-toolbar button').forEach((button) => {
      button.addEventListener('click', () => {
        const toolbar = button.closest('.chart-toolbar');
        const canvas = document.getElementById(toolbar.dataset.target);
        const state = chartState.get(canvas) || {zoom: 1};
        if (button.dataset.action === 'zoom-in') state.zoom = Math.min(3, state.zoom * 1.25);
        if (button.dataset.action === 'zoom-out') state.zoom = Math.max(0.45, state.zoom / 1.25);
        if (button.dataset.action === 'reset') state.zoom = 1;
        chartState.set(canvas, state);
        redraw(canvas);
      });
    });
