import { PIPELINE_STAGES } from './compute/protocol';

export function App() {
  return (
    <main className="shell">
      <header>
        <p className="eyebrow">Muon tomography observatory</p>
        <h1>UMBRA</h1>
        <p className="lede">
          Build a pyramid, hide a chamber, place detectors — then do the inverse
          problem for real: reconstruct the density, decide whether the void
          exists, and choose where the next detector should go.
        </p>
      </header>

      <section aria-labelledby="status">
        <h2 id="status">Repository status</h2>
        <p>
          This is the front-end scaffold. The compute core is a Rust workspace
          compiled to WebAssembly; the interface below lists the tasks the
          <code>ComputeBus</code> will orchestrate. Nothing is wired up yet — the
          scaffold exists so the worker protocol, the store and the renderer can
          be built against a fixed contract.
        </p>
        <ol className="stages">
          {PIPELINE_STAGES.map((stage) => (
            <li key={stage.kind}>
              <span className="kind">{stage.kind}</span>
              <span className="detail">{stage.detail}</span>
            </li>
          ))}
        </ol>
      </section>

      <footer>
        <p>
          Spec: <code>source/UMBRA_DESIGN_SPEC.md</code>. Docs:{' '}
          <code>docs/ARCHITECTURE.md</code>, <code>docs/ROADMAP.md</code>.
        </p>
      </footer>
    </main>
  );
}
