'use strict';
const $ = (id) => document.getElementById(id);
let current = null, metadata = null, busy = false;
const advisorLabel = (mode) => ({offline: 'offline heuristic', cached: 'cached Jev', live: 'hosted Jev'})[mode] || 'advisor';
const display = (value) => value === null || value === undefined ? 'Not inspected' : typeof value === 'object' ? JSON.stringify(value) : String(value);
const numeric = (value, digits = 2) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : '—';
function status(message, error = false) { $('status').textContent = message; $('status').classList.toggle('error', error); }
async function api(route, data) {
  const response = await fetch(route, {headers: {'Content-Type': 'application/json'}, ...(data === undefined ? {} : {method: 'POST', body: JSON.stringify(data)})});
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || `Request failed (${response.status})`);
  return value;
}
function setBusy(value) {
  busy = value;
  $('inject').disabled = value;
  $('run').disabled = value || !current || current.done || (metadata && !metadata.policies.find(p => p.id === $('policy').value)?.available);
  document.querySelectorAll('[data-action]').forEach(button => { button.disabled = value || !current || current.done || (button.dataset.action === 'jev' && !current.observation.evidence?.logs); });
  $('policy').disabled = value;
}
async function operation(task) {
  setBusy(true);
  try { await task(); } catch (error) { status(error.message, true); }
  finally { setBusy(false); }
}
function node(tag, text, cls) { const result = document.createElement(tag); if (text !== undefined) result.textContent = text; if (cls) result.className = cls; return result; }
function evidenceSummary(title, value, inspected) {
  if (value === null || value === undefined) return inspected ? 'Invalidated after repair · see timeline' : 'Not inspected';
  if (typeof value !== 'object') return display(value);
  if (title === 'Configuration') return `${value.valid ? 'Credentials valid' : 'Invalid credentials'} · ${value.connection_release_enabled ? 'release enabled' : 'release disabled'}`;
  if (title === 'Connections') return `${value.active}/${value.capacity} sessions · ${value.exhausted ? 'exhausted' : 'available'}`;
  if (title === 'Dependency') return `${value.reachable ? 'Reachable' : 'Unreachable'} · circuit ${value.circuit_open ? 'open' : 'closed'}`;
  return display(value);
}
function healthValue(observation) { const health = observation.health; return typeof health === 'object' ? Boolean(health.healthy ?? health.service_healthy) : Boolean(health); }
function render(session) {
  current = session;
  const o = session.observation, m = session.metrics, e = o.evidence || {};
  const healthy = healthValue(o);
  $('session').textContent = `SESSION ${session.id.slice(0, 8).toUpperCase()}`;
  $('health').textContent = m.recovered ? 'Recovered' : healthy ? 'Healthy · unverified' : 'Degraded';
  $('health').className = healthy ? 'good' : 'bad';
  $('termination').textContent = session.done ? `Terminated: ${m.termination || 'finished'}` : 'Independent verification pending';
  $('downtime').textContent = m.downtime_ticks ?? 0;
  $('deadline').textContent = `${o.tick} elapsed · ${o.remaining_ticks} ticks remaining`;
  $('actions').textContent = m.actions ?? 0;
  $('tool-cost').textContent = numeric(m.tool_cost_units);
  $('api-cost').textContent = `Hosted API cost: $${numeric(m.api_cost_usd || 0, 6)}`;
  $('streak').textContent = `HEALTH STREAK ${o.healthy_streak ?? 0}`;
  $('logs').textContent = e.logs ? display(e.logs) : session.done ? 'This controller did not collect a log excerpt.' : 'No logs collected. Select “Inspect logs” to gather evidence.';
  $('diagnostics').replaceChildren();
  [['Configuration', e.config, 'config'], ['Connections', e.connections, 'connections'], ['Dependency', e.dependency, 'dependency']].forEach(([title, value, action]) => {
    const row = node('div', undefined, 'diagnostic'); row.append(node('span', title), node('span', evidenceSummary(title, value, (o.action_counts?.[action] || 0) > 0))); $('diagnostics').append(row);
  });
  $('hypotheses').replaceChildren();
  const diagnosis = e.diagnosis;
  const probabilities = diagnosis?.probabilities || diagnosis?.scores;
  if (diagnosis?.available && probabilities) {
    Object.entries(probabilities).sort((a, b) => b[1] - a[1]).forEach(([cause, score]) => {
      const row = node('div', undefined, 'hypothesis'), head = node('div', undefined, 'hypothesishead');
      head.append(node('span', cause.toUpperCase()), node('span', `${numeric(score * 100, 0)}%`));
      const meter = node('meter', `${numeric(score * 100, 0)}%`, 'meter'); meter.min = 0; meter.max = 1; meter.value = score; meter.setAttribute('aria-label', `${cause} estimated likelihood`);
      row.append(head, meter); $('hypotheses').append(row);
    });
    $('advisor-origin').textContent = `SOURCE: ${diagnosis.source || session.advisor_source} · ${diagnosis.cache_hit ? 'CACHE HIT' : session.advisor_mode === 'live' ? 'HOSTED CALL' : 'LOCAL COMPUTATION'}`;
  } else {
    $('hypotheses').append(node('p', diagnosis ? 'Diagnosis unavailable. Check your TypeSafe key, network or cache if using Jev.' : session.done ? 'This controller did not request an advisor diagnosis.' : `Collect logs, then consult ${advisorLabel(session.advisor_mode)}.`, 'empty'));
    $('advisor-origin').textContent = diagnosis ? 'DIAGNOSIS UNAVAILABLE' : session.done ? 'NO ADVISOR CONSULTATION' : `ADVISOR: ${advisorLabel(session.advisor_mode).toUpperCase()}`;
  }
  $('timeline').replaceChildren();
  const trace = session.trace || [];
  $('trace-count').textContent = `${trace.length} ACTIONS`;
  if (!trace.length) { const row = node('tr'), cell = node('td', 'No actions yet. Investigate manually or run a controller.', 'empty'); cell.colSpan = 5; row.append(cell); $('timeline').append(row); }
  const decisions = session.decisions || [];
  trace.forEach((event, index) => {
    const row = node('tr');
    const matched = decisions.find(decision => decision.trace_index === index);
    const after = event.observation_after || event.observation || event.after || {};
    const outcome = event.outcome ?? after.last_outcome ?? event.result ?? 'Action executed';
    [event.tick ?? event.tick_after ?? after.tick ?? '—', event.action, outcome?.message || display(outcome), numeric(event.reward), matched?.reason || 'Manual tool invocation'].forEach(value => row.append(node('td', value)));
    $('timeline').append(row);
  });
  status(session.done ? (m.recovered ? 'Recovery verified by sustained simulator health checks.' : `Incident ended without verified recovery (${m.termination || 'terminated'}).`) : `Incident active. ${o.remaining_ticks} simulated ticks remain.`);
  setBusy(busy);
}
$('inject-form').addEventListener('submit', event => {
  event.preventDefault();
  operation(async () => {
    const faults = Array.from(document.querySelectorAll('input[name=fault]:checked')).map(input => input.value);
    render(await api('/api/incidents', {faults, wording: $('wording').value, seed: Number($('seed').value)}));
  });
});
document.querySelectorAll('[data-action]').forEach(button => button.addEventListener('click', () => operation(async () => {
  status(`Executing ${button.dataset.action}…`);
  render(await api(`/api/incidents/${current.id}/step`, {action: button.dataset.action}));
})));
$('run').addEventListener('click', () => operation(async () => {
  status('Controller investigating and verifying recovery…');
  render(await api(`/api/incidents/${current.id}/run`, {policy: $('policy').value}));
}));
$('policy').addEventListener('change', () => {
  const available = metadata?.policies.find(policy => policy.id === $('policy').value)?.available;
  $('policy-note').textContent = available ? 'The controller uses the same tools and sustained-health verifier.' : 'No trained checkpoint found. Run the v2 benchmark to enable this controller.';
  setBusy(busy);
});
async function report() {
  try {
    const data = await api('/api/report');
    $('report-json').textContent = JSON.stringify(data, null, 2);
    $('report-summary').replaceChildren();
    const notice = node('div', undefined, 'reportnotice');
    notice.append(node('strong', 'PROVIDER PROVENANCE / '), node('span', data.advisor?.label || display(data.advisor_provenance || data.provenance || data.advisor || data.advisor_mode || 'Review the provider metadata in the full report.')));
    $('report-summary').append(notice);
    const source = data.results || data.aggregate || data.policies || data.summary || {};
    const entries = Array.isArray(source) ? source.map((value, i) => [value.policy || value.name || String(i), value]) : Object.entries(source);
    const rows = entries.filter(([, value]) => value && typeof value === 'object');
    $('report-table').replaceChildren();
    if (rows.length) {
      const table = node('table'), head = node('thead'), header = node('tr'), body = node('tbody');
      ['Policy', 'Recovery rate', 'Mean actions', 'Downtime ticks', 'Tool cost'].forEach(title => header.append(node('th', title)));
      head.append(header);
      rows.forEach(([name, value]) => {
        const v = value.overall || value.metrics || value, row = node('tr');
        const rate = v.recovery_rate ?? v.recovered_rate ?? v.success_rate;
        const advisorName = data.advisor?.mode === 'offline' ? 'offline advisor' : 'Jev advisor';
        const labels = { checklist: 'Troubleshooting checklist', greedy: 'Repair-first baseline', random: 'Random actions', rl_without_jev: 'RL without advisor', rl_with_jev: `RL with ${advisorName}`, rl_with_corrupted_advisor: `RL with corrupted ${advisorName}` };
        [labels[name] || name.replaceAll('_', ' '), rate === undefined ? 'See full report' : `${numeric(rate * 100, 1)}%`, numeric(v.actions ?? v.mean_actions), numeric(v.downtime_ticks ?? v.mean_downtime_ticks), numeric(v.tool_cost_units ?? v.mean_tool_cost_units)].forEach(text => row.append(node('td', text)));
        body.append(row);
      });
      table.append(head, body); $('report-table').append(table);
    } else $('report-table').append(node('p', 'This report uses a detailed schema. Inspect the reproducibility report below.', 'empty'));
  } catch (error) {
    $('report-summary').replaceChildren(node('p', error.message, 'reportnotice')); $('report-json').textContent = 'No report loaded.'; $('report-table').replaceChildren();
  }
}
function selectTab(tab) {
  ['incident', 'report'].forEach(name => { const selected = name === tab; $(`${name}-tab`).setAttribute('aria-selected', selected); $(`${name}-tab`).tabIndex = selected ? 0 : -1; $(`${name}-panel`).hidden = !selected; });
  if (tab === 'report') report();
}
['incident', 'report'].forEach((name, index) => {
  $(`${name}-tab`).addEventListener('click', () => selectTab(name));
  $(`${name}-tab`).addEventListener('keydown', event => { if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) { event.preventDefault(); const target = event.key === 'Home' ? 'incident' : event.key === 'End' ? 'report' : ['report', 'incident'][index]; selectTab(target); $(`${target}-tab`).focus(); } });
});
$('refresh-report').addEventListener('click', report);
api('/api/meta').then(data => {
  metadata = data;
  $('advisor-mode').textContent = advisorLabel(data.advisor_mode);
  $('advisor-mode-detail').textContent = data.hosted_api_enabled ? 'Consulting the advisor can send collected simulated logs to TypeSafe.' : 'This mode makes no hosted requests.';
  $('advisor-origin').textContent = `ADVISOR: ${advisorLabel(data.advisor_mode).toUpperCase()}`;
  for (const option of $('policy').options) { const available = data.policies.find(p => p.id === option.value)?.available; if (!available) option.textContent += ' (not trained)'; }
  status('Local lab ready. Inject an incident to begin.'); setBusy(false);
}).catch(error => status(error.message, true));
