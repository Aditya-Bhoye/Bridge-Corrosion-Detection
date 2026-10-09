import React, {useEffect, useState} from 'react';
import {Layers, Image as ImageIcon} from 'lucide-react';

const calm = () => typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;

// Animates a number from 0 up to `target` with an ease-out curve.
export function useCountUp(target, duration = 900) {
  const [value, setValue] = useState(calm() && Number.isFinite(target) ? target : 0);
  useEffect(() => {
    if (!Number.isFinite(target)) { setValue(0); return; }
    if (calm()) { setValue(target); return; }
    let frame, start;
    const tick = (t) => {
      start ??= t;
      const p = Math.min(1, (t - start) / duration);
      setValue(target * (1 - Math.pow(1 - p, 3)));
      if (p < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [target, duration]);
  return value;
}

export function Stat({label, value, format = (v) => Math.round(v), note, icon: Icon, delay = 0}) {
  const shown = useCountUp(value);
  return (
    <div className="stat rise" style={{animationDelay: `${delay}ms`}}>
      <div className="stat-top"><span>{label}</span><span className="stat-icon"><Icon size={17}/></span></div>
      <strong>{value === null ? '—' : format(shown)}</strong>
      <small>{note}</small>
    </div>
  );
}

export function Coverage({value}) {
  return <strong>{useCountUp(value).toFixed(1)}<span>%</span></strong>;
}

// Suspension bridge line drawing with a scanning beam, used as the upload illustration.
export function BridgeArt() {
  const hangers = [104, 128, 152, 176, 224, 248, 272, 296];
  return (
    <svg className="bridge-art" viewBox="0 0 400 150" aria-hidden="true">
      <defs>
        <linearGradient id="beam" x1="0" x2="1">
          <stop offset="0" stopColor="currentColor" stopOpacity="0"/>
          <stop offset=".5" stopColor="currentColor" stopOpacity=".35"/>
          <stop offset="1" stopColor="currentColor" stopOpacity="0"/>
        </linearGradient>
      </defs>
      <path className="water" d="M0 138 Q25 133 50 138 T100 138 T150 138 T200 138 T250 138 T300 138 T350 138 T400 138"/>
      <path className="draw deck" d="M10 112 H390"/>
      <path className="draw tower" d="M80 125 V28 M320 125 V28 M72 30 H88 M312 30 H328"/>
      <path className="draw cable" d="M10 70 Q45 52 80 32 Q200 128 320 32 Q355 52 390 70"/>
      {hangers.map((x, i) => {
        const t = (x - 80) / 240, y = (1 - t) * (1 - t) * 32 + 2 * (1 - t) * t * 128 + t * t * 32;
        return <path key={x} className="draw hanger" style={{animationDelay: `${0.9 + i * 0.05}s`}} d={`M${x} ${y.toFixed(1)} V112`}/>;
      })}
      <circle className="rust r1" cx="145" cy="112" r="4"/>
      <circle className="rust r2" cx="255" cy="112" r="3"/>
      <circle className="rust r3" cx="320" cy="78" r="3"/>
      <rect className="beam" x="-60" y="10" width="60" height="120" fill="url(#beam)"/>
    </svg>
  );
}

const BANDS = [['Low', 0, 10], ['Moderate', 10, 25], ['High', 25, 100]];

export function BandScale({coverage}) {
  // Stretch 0–25% across most of the width so the low and moderate bands stay readable.
  const active = coverage >= 25 ? 'High' : coverage >= 10 ? 'Moderate' : 'Low';
  const position = (v) => v <= 25 ? (v / 25) * 75 : 75 + ((v - 25) / 75) * 25;
  return (
    <div className="band-scale" role="img" aria-label={`Coverage ${coverage.toFixed(1)}% on a scale with bands Low under 10%, Moderate 10 to 25%, High 25% and over`}>
      <div className="band-track">
        {BANDS.map(([name, from, to]) => (
          <span key={name} className={'band ' + name.toLowerCase() + (name === active ? ' active' : '')} style={{left: `${position(from)}%`, width: `${position(to) - position(from)}%`}}/>
        ))}
        <i className="band-marker" style={{'--to': `${position(coverage)}%`}}/>
      </div>
      <div className="band-names">
        {BANDS.map(([name, from, to]) => <span key={name} className={name.toLowerCase()} style={{left: `${position(from)}%`, width: `${position(to) - position(from)}%`}}>{name}</span>)}
      </div>
      <div className="band-ticks">
        {[0, 10, 25, 100].map(t => <span key={t} style={{left: `${position(t)}%`}}>{t}%</span>)}
      </div>
    </div>
  );
}

export function Viewer({src, mask, overlay, setOverlay, opacity, setOpacity, scanning}) {
  return (
    <div className="viewer">
      {mask && (
        <div className="viewer-bar">
          <div className="segmented" role="tablist" aria-label="Image layer">
            <button role="tab" aria-selected={!overlay} className={!overlay ? 'on' : ''} onClick={() => setOverlay(false)}><ImageIcon size={15}/>Photo</button>
            <button role="tab" aria-selected={overlay} className={overlay ? 'on' : ''} onClick={() => setOverlay(true)}><Layers size={15}/>Corrosion overlay</button>
          </div>
          {overlay && (
            <label className="opacity">
              Opacity
              <input type="range" min="0.2" max="1" step="0.05" value={opacity} onChange={e => setOpacity(+e.target.value)}/>
            </label>
          )}
        </div>
      )}
      <div className="stage-wrap">
        <div className={'stage' + (scanning ? ' scanning' : '')}>
          <img src={src} alt="Bridge surface being inspected"/>
          {mask && overlay && <img className="layer reveal" src={mask} style={{'--o': opacity, opacity}} alt="Predicted corrosion mask"/>}
          {scanning && <><span className="scan-grid"/><span className="scan-beam"/></>}
        </div>
      </div>
    </div>
  );
}
