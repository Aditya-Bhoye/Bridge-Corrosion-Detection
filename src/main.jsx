import React, {useEffect, useRef, useState} from 'react';
import {createRoot} from 'react-dom/client';
import {ShieldCheck, ScanLine, History, UploadCloud, ImagePlus, Activity, Trash2, Download, Menu, X, CheckCircle2, AlertTriangle, RotateCcw, ChevronRight, Gauge, MonitorSmartphone} from 'lucide-react';
import {validateFile, validateResult} from './detection';
import {Stat, BandScale, Viewer, BridgeArt, Coverage} from './ui';
import {modelAvailable, loadModel, detectInBrowser, onProgress} from './inference/client';
import './style.css';

function modelStatus(model, progress) {
  if (model.state === 'loading') return progress < 1 ? `Preparing ${Math.round(progress * 100)}%` : 'Preparing…';
  return {checking: 'Starting…', available: 'Ready', ready: 'Ready', missing: 'Unavailable', error: 'Unavailable'}[model.state];
}

// Static hosts cannot send cross-origin isolation headers, so a service worker adds them.
// It only takes effect after one reload; sessionStorage stops a reload loop if isolation still fails.
if (import.meta.env.PROD && !window.crossOriginIsolated && window.isSecureContext && 'serviceWorker' in navigator) {
  navigator.serviceWorker.register(`${import.meta.env.BASE_URL}coi-sw.js`).then(reg => {
    let tried = false;
    try { tried = sessionStorage.getItem('coi-reloaded') === '1'; sessionStorage.setItem('coi-reloaded', '1'); } catch {}
    if (!tried && (reg.active || reg.installing || reg.waiting)) {
      const reload = () => window.location.reload();
      navigator.serviceWorker.controller ? reload() : navigator.serviceWorker.addEventListener('controllerchange', reload, {once: true});
    }
  }).catch(() => {});
}

function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem('bridgeguard-history') || '[]')
      .filter(r => r && !r.demo && typeof r.name === 'string' && Number.isFinite(r.coverage)).slice(0, 50);
  } catch { return []; }
}

function App() {
  const [page, setPage] = useState('Inspect');
  const [mobile, setMobile] = useState(false);
  const [model, setModel] = useState({state: 'checking'});
  const [progress, setProgress] = useState(0);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [drag, setDrag] = useState(false);
  const [overlay, setOverlay] = useState(true);
  const [opacity, setOpacity] = useState(0.7);
  const [records, setRecords] = useState(loadHistory);
  const input = useRef(), active = useRef(true);

  useEffect(() => { active.current = true; return () => { active.current = false; }; }, []);
  useEffect(() => {
    modelAvailable().then(ok => setModel(m => m.state === 'checking' ? {state: ok ? 'available' : 'missing'} : m));
    return onProgress((value, type) => { if (type === 'progress') setProgress(value); });
  }, []);
  useEffect(() => {
    try { localStorage.setItem('bridgeguard-history', JSON.stringify(records)); }
    catch { setError('Browser storage is full. History will last only for this session.'); }
  }, [records]);
  useEffect(() => {
    if (!file) { setPreview(''); return; }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  function choose(f) {
    if (!f || busy) return;
    const message = validateFile(f);
    setError(message);
    if (!message) { setFile(f); setResult(null); prepareModel().catch(() => {}); }
  }

  // Start the one-time model download as soon as someone picks a photo, so it overlaps with their next click.
  function prepareModel() {
    if (model.state !== 'available' && model.state !== 'error') return Promise.resolve();
    setModel({state: 'loading'});
    return loadModel().then(
      backend => { if (active.current) setModel({state: 'ready', backend}); },
      e => { if (active.current) setModel({state: e.message === 'MODEL_MISSING' ? 'missing' : 'error'}); throw e; },
    );
  }

  async function analyze() {
    if (!file || busy) return;
    setBusy(true); setError(''); setResult(null);
    try {
      if (model.state === 'missing') throw new Error('MODEL_MISSING');
      await prepareModel();
      const {coverage, mask, backend, ms} = await detectInBrowser(file);
      const data = {...validateResult({coverage}), mask, backend, ms};
      if (!active.current) return;
      setResult(data);
      setOverlay(true);
      setRecords(rows => [{...data, mask: undefined, id: crypto.randomUUID(), name: file.name, date: new Date().toISOString()}, ...rows].slice(0, 50));
    } catch (e) {
      if (active.current) setError(e.message === 'MODEL_MISSING' ? 'Analysis is unavailable because the model file has not been deployed with this site yet.' : e.message);
    } finally {
      if (active.current) setBusy(false);
    }
  }

  function reset() { setFile(null); setResult(null); setError(''); }

  function exportHistory() {
    const blob = new Blob([JSON.stringify(records, null, 2)], {type: 'application/json'});
    const url = URL.createObjectURL(blob), a = document.createElement('a');
    a.href = url; a.download = 'bridgeguard-inspections.json'; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function navigate(p) { setPage(p); setMobile(false); window.scrollTo({top: 0}); }

  const high = records.filter(r => r.severity === 'High').length;
  const avg = records.length ? records.reduce((s, r) => s + r.coverage, 0) / records.length : null;
  const tone = {ready: 'ready', available: 'ready', loading: 'checking', checking: 'checking', missing: 'offline', error: 'offline'}[model.state];
  const statusText = modelStatus(model, progress);

  return (
    <div className="app">
      <aside className={mobile ? 'sidebar open' : 'sidebar'}>
        <a className="brand" href="#" onClick={e => { e.preventDefault(); navigate('Inspect'); }}>
          <span className="brand-icon"><ShieldCheck size={22}/><i/></span>BridgeGuard
        </a>
        <nav>
          {[['Inspect', ScanLine], ['History', History]].map(([p, Icon]) => (
            <button key={p} className={page === p ? 'nav-item selected' : 'nav-item'} aria-current={page === p ? 'page' : undefined} onClick={() => navigate(p)}>
              <Icon size={18}/>{p === 'Inspect' ? 'New inspection' : 'Inspection history'}
              {p === 'History' && <span className="nav-count">{records.length}</span>}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className={'model-card ' + tone}>
            <div className="model-card-top"><MonitorSmartphone size={16}/> Runs in your browser</div>
            <strong><i/>{statusText}</strong>
            <p>Free to use. Photos never leave this device.</p>
          </div>
        </div>
      </aside>
      {mobile && <button className="backdrop" aria-label="Close navigation" onClick={() => setMobile(false)}/>}

      <main>
        <header>
          <button className="mobile-toggle" aria-label={mobile ? 'Close navigation' : 'Open navigation'} onClick={() => setMobile(!mobile)}>{mobile ? <X/> : <Menu/>}</button>
          <div>
            <h1>{page === 'Inspect' ? 'Corrosion inspection' : 'Inspection history'}</h1>
            <p>{page === 'Inspect' ? 'Upload a bridge photo to measure how much of the surface shows corrosion.' : 'Measurements saved in this browser. Photos are never stored.'}</p>
          </div>
          <span className={'status ' + tone}><i/>{statusText}</span>
        </header>

        <div className="content">
          {page === 'Inspect' && (
            <>
              <section className="workspace">
                <div className="panel image-panel rise">
                  <div className="panel-heading">
                    <div><span className="step">1</span><h2>Photo</h2></div>
                  </div>

                  {preview ? (
                    <Viewer src={preview} mask={result?.mask} overlay={overlay} setOverlay={setOverlay} opacity={opacity} setOpacity={setOpacity} scanning={busy}/>
                  ) : (
                    <div
                      className={'dropzone' + (drag ? ' drag' : '')}
                      role="button" tabIndex={0}
                      onClick={() => input.current.click()}
                      onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.current.click(); } }}
                      onDragOver={e => { e.preventDefault(); setDrag(true); }}
                      onDragLeave={() => setDrag(false)}
                      onDrop={e => { e.preventDefault(); setDrag(false); choose(e.dataTransfer.files[0]); }}
                    >
                      <BridgeArt/>
                      <div className="upload-icon"><UploadCloud size={22}/></div>
                      <h3>Drop a photo of the bridge surface</h3>
                      <p>or <u>browse your files</u></p>
                      <small>JPG, PNG or WebP · up to 10 MB · close, well-lit shots work best</small>
                    </div>
                  )}
                  <input ref={input} type="file" accept="image/jpeg,image/png,image/webp" hidden onChange={e => { choose(e.target.files[0]); e.target.value = ''; }}/>

                  {file && (
                    <div className="file-info">
                      <ImagePlus size={16}/><span>{file.name}</span><small>{(file.size / 1024 / 1024).toFixed(2)} MB</small>
                      <button className="link" disabled={busy} onClick={() => input.current.click()}>Change</button>
                      <button className="link" disabled={busy} onClick={reset} aria-label="Remove photo"><RotateCcw size={14}/></button>
                    </div>
                  )}
                  {model.state === 'missing' && <div className="notice warn"><AlertTriangle size={16}/>Analysis is unavailable because the model file has not been deployed with this site yet.</div>}
                  {model.state === 'loading' && progress < 1 && (
                    <div className="notice download">
                      <Download size={16}/>
                      <div><span>Getting ready for the first analysis (about 31 MB, one time only).</span><div className="progress"><i style={{width: `${progress * 100}%`}}/></div></div>
                    </div>
                  )}
                  {error && <p className="error" role="alert">{error}</p>}
                  <button className="primary analyze" disabled={!file || busy} onClick={analyze}>
                    {busy ? <span className="spinner"/> : <ScanLine size={18}/>}
                    {busy ? 'Analysing photo…' : result ? 'Analyse again' : 'Analyse photo'}
                  </button>
                </div>

                <div className="panel result-panel rise" style={{animationDelay: '80ms'}} aria-live="polite">
                  <div className="panel-heading"><div><span className="step">2</span><h2>Assessment</h2></div></div>
                  {!result ? (
                    <div className="empty-result rise">
                      {busy ? <span className="spinner big"/> : <div className="target-icon"><i/><i/><ScanLine size={30}/></div>}
                      <h3>{busy ? 'Inspecting your photo' : 'Ready when you are'}</h3>
                      <p>{busy ? 'Your results will appear here.' : 'Upload a photo and run an inspection to see corrosion coverage.'}</p>
                    </div>
                  ) : (
                    <div className="result rise">
                      <div className="measurement">
                        <div>
                          <small>Corrosion coverage</small>
                          <Coverage value={result.coverage}/>
                        </div>
                        <span className={'badge ' + result.severity.toLowerCase()}>{result.severity}</span>
                      </div>
                      <BandScale coverage={result.coverage}/>
                      <p className="fine">Coverage is the share of the photo classified as corrosion. It does not measure structural condition.<small className="ran-on">Analysed on this device in {(result.ms / 1000).toFixed(1)} s. Your photo was not uploaded.</small></p>
                      <div className="saved"><CheckCircle2 size={16}/>Saved to inspection history</div>
                    </div>
                  )}
                </div>
              </section>

              <section className="stats">
                <Stat label="Inspections" value={records.length} note="Saved in this browser" icon={ScanLine}/>
                <Stat label="High coverage" value={high} note="25% of the photo or more" icon={Activity} delay={80}/>
                <Stat label="Average coverage" value={avg} format={v => `${v.toFixed(1)}%`} note="Across all inspections" icon={Gauge} delay={160}/>
              </section>
            </>
          )}

          <section className="panel history-panel rise" style={{animationDelay: '200ms'}}>
            <div className="panel-heading">
              <div><h2>{page === 'History' ? 'All inspections' : 'Recent inspections'}</h2></div>
              <div className="history-actions">
                {page !== 'History' && records.length > 5 && <button className="subtle" onClick={() => navigate('History')}>View all <ChevronRight size={15}/></button>}
                <button className="subtle" disabled={!records.length} onClick={exportHistory}><Download size={15}/>Export JSON</button>
              </div>
            </div>
            {!records.length ? (
              <div className="history-empty"><History size={22}/><span>No inspections yet. Your first result will appear here.</span></div>
            ) : (
              <div className="table-wrap">
                <table>
                  <thead><tr><th>Photo</th><th>Date</th><th>Coverage</th><th>Level</th><th><span className="sr-only">Actions</span></th></tr></thead>
                  <tbody>
                    {(page === 'History' ? records : records.slice(0, 5)).map(r => (
                      <tr key={r.id}>
                        <td><div className="table-name"><ImagePlus size={16}/><span>{r.name}</span></div></td>
                        <td>{new Date(r.date).toLocaleString()}</td>
                        <td className="num">{r.coverage.toFixed(1)}%</td>
                        <td><span className={'badge ' + r.severity.toLowerCase()}>{r.severity}</span></td>
                        <td><button className="delete" aria-label={`Delete inspection ${r.name}`} onClick={() => setRecords(rows => rows.filter(x => x.id !== r.id))}><Trash2 size={16}/></button></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
          <footer><ShieldCheck size={14}/>BridgeGuard measures visible surface corrosion in photos. It supports, and does not replace, a qualified bridge inspector.</footer>
        </div>
      </main>
    </div>
  );
}

createRoot(document.getElementById('root')).render(<App/>);
