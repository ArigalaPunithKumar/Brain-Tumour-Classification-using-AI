import { useEffect, useMemo, useState } from "react";
import { Link, Navigate, NavLink, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { api } from "./api";

const navItems = [
  { to: "/", label: "Home" },
  { to: "/about", label: "About" },
  { to: "/prediction", label: "AI Analysis", private: true },
];

function App() {
  const [user, setUser] = useState(null);
  const [checkingAuth, setCheckingAuth] = useState(true);

  useEffect(() => {
    api.me()
      .then((data) => setUser(data.authenticated ? data.user : null))
      .catch(() => setUser(null))
      .finally(() => setCheckingAuth(false));
  }, []);

  const login = (nextUser) => setUser(nextUser);

  const logout = async () => {
    try {
      await api.logout();
    } finally {
      setUser(null);
    }
  };

  if (checkingAuth) return <Splash />;

  return (
    <div className="app-shell">
      <Header user={user} onLogout={logout} />
      <main>
        <Routes>
          <Route path="/" element={<Home user={user} />} />
          <Route path="/about" element={<About />} />
          <Route path="/login" element={<AuthPage mode="login" onLogin={login} />} />
          <Route path="/register" element={<AuthPage mode="register" onLogin={login} />} />
          <Route path="/prediction" element={
            <ProtectedRoute user={user}><Prediction /></ProtectedRoute>
          } />
          <Route path="/dashboard" element={
            <ProtectedRoute user={user}><Dashboard user={user} /></ProtectedRoute>
          } />
          <Route path="/admin" element={
            <ProtectedRoute user={user} adminOnly><AdminDashboard user={user} /></ProtectedRoute>
          } />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
      <Footer />
    </div>
  );
}

function Header({ user, onLogout }) {
  const navigate = useNavigate();
  return (
    <header className="site-header">
      <Link to="/" className="brand">
        <span className="brand-mark"><span /></span>
        <span><strong>NeuroVision</strong><small>AI MRI ASSIST</small></span>
      </Link>

      <nav className="desktop-nav">
        {navItems.filter((item) => !item.private || user).map((item) => (
          <NavLink key={item.to} to={item.to} className={({ isActive }) => isActive ? "active" : ""}>
            {item.label}
          </NavLink>
        ))}
        {user?.role === "admin" && <NavLink to="/admin">Admin</NavLink>}
      </nav>

      <div className="header-actions">
        {user ? (
          <>
            <button className="avatar-button" onClick={() => navigate(user.role === "admin" ? "/admin" : "/dashboard")}>
              {user.name?.charAt(0)?.toUpperCase() || "U"}
            </button>
            <button className="button ghost compact" onClick={onLogout}>Logout</button>
          </>
        ) : (
          <Link to="/login" className="button primary compact">Sign in</Link>
        )}
      </div>
    </header>
  );
}

function Home({ user }) {
  return (
    <div>
      <section className="hero">
        <div className="hero-glow glow-one" />
        <div className="hero-glow glow-two" />
        <div className="hero-copy">
          <div className="eyebrow"><span className="pulse-dot" /> AI-ASSISTED MRI ANALYSIS</div>
          <h1>See the scan.<br /><em>Understand the pattern.</em></h1>
          <p className="hero-text">
            A deep-learning workflow for brain MRI relevance detection, tumour classification,
            and predicted tumour-region segmentation.
          </p>
          <div className="hero-actions">
            <Link to={user ? "/prediction" : "/register"} className="button primary large">
              {user ? "Start MRI analysis" : "Get started"}
              <span>→</span>
            </Link>
            <Link to="/about" className="button ghost large">Explore the pipeline</Link>
          </div>
          <div className="trust-row">
            <span>MobileNet</span><i /> <span>U-Net</span><i /> <span>ResNet34</span><i /> <span>Flask API</span>
          </div>
        </div>

        <div className="brain-visual" aria-hidden="true">
          <div className="scan-ring ring-a" />
          <div className="scan-ring ring-b" />
          <div className="scan-ring ring-c" />
          <div className="mri-orb">
            <div className="mri-brain">
              <div className="brain-left" />
              <div className="brain-right" />
              <div className="tumour-node" />
            </div>
          </div>
          <div className="data-chip chip-one"><b>01</b><span>MRI VALID</span></div>
          <div className="data-chip chip-two"><b>02</b><span>CLASSIFY</span></div>
          <div className="data-chip chip-three"><b>03</b><span>SEGMENT</span></div>
        </div>
      </section>

      <section className="section">
        <div className="section-heading">
          <div><span className="eyebrow">FROM IMAGE TO INSIGHT</span><h2>One scan. Three intelligent stages.</h2></div>
          <p>The interface keeps the workflow simple while the backend handles the model inference.</p>
        </div>
        <div className="stage-grid">
          <Stage number="01" title="MRI relevance" text="Filters out images that do not match the expected brain MRI input." />
          <Stage number="02" title="Tumour classification" text="Uses a trained MobileNet classifier to identify tumour presence." />
          <Stage number="03" title="Region segmentation" text="Uses U-Net with a ResNet34 encoder to produce a predicted mask." />
        </div>
      </section>

      <section className="disclaimer">
        <span>ⓘ</span>
        <p><strong>Academic project.</strong> This application is for software/research demonstration only and is not a replacement for professional medical diagnosis.</p>
      </section>
    </div>
  );
}

function Stage({ number, title, text }) {
  return <article className="stage-card">
    <span className="stage-number">{number}</span>
    <div className="stage-icon"><span /></div>
    <h3>{title}</h3>
    <p>{text}</p>
  </article>;
}

function About() {
  return <section className="page-section narrow">
    <div className="eyebrow">ABOUT THE PROJECT</div>
    <h1>AI-assisted brain MRI analysis.</h1>
    <p className="lead">
      NeuroVision turns the original Life Care Flask workflow into a clean, deployable architecture:
      React on Vercel, Flask/PyTorch on Render, and MySQL on Aiven.
    </p>
    <div className="feature-grid">
      <InfoCard title="Relevance detection" text="A fine-tuned MobileNet model checks whether an uploaded image is a relevant brain MRI." />
      <InfoCard title="Classification" text="A second MobileNet model predicts whether a tumour is detected." />
      <InfoCard title="Segmentation" text="A U-Net model with a ResNet34 encoder generates a predicted tumour mask." />
      <InfoCard title="Secure architecture" text="Credentials stay in the Flask API and database; the Vercel frontend never receives database secrets." />
    </div>
  </section>;
}

function InfoCard({ title, text }) {
  return <article className="info-card"><div className="mini-line" /><h3>{title}</h3><p>{text}</p></article>;
}

function AuthPage({ mode, onLogin }) {
  const isLogin = mode === "login";
  const navigate = useNavigate();
  const [form, setForm] = useState(isLogin
    ? { email: "", password: "", role: "user" }
    : { name: "", email: "", password: "", confirm_password: "", age: "", gender: "O", mobile: "" });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const update = (key, value) => setForm((current) => ({ ...current, [key]: value }));

  const submit = async (event) => {
    event.preventDefault();
    setError("");
    setBusy(true);
    try {
      const data = isLogin ? await api.login(form) : await api.register(form);
      if (isLogin) {
        onLogin(data.user);
        navigate(data.user.role === "admin" ? "/admin" : "/dashboard");
      } else {
        navigate("/login", { state: { registered: true } });
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  return <section className="auth-layout">
    <div className="auth-art">
      <div className="auth-orbit orbit-1" /><div className="auth-orbit orbit-2" />
      <div className="auth-brain"><div /><div /><span /></div>
      <div className="auth-caption"><span className="pulse-dot" /> Secure patient workflow</div>
    </div>
    <div className="auth-panel">
      <span className="eyebrow">{isLogin ? "WELCOME BACK" : "CREATE ACCOUNT"}</span>
      <h1>{isLogin ? "Continue to your workspace." : "Start your analysis journey."}</h1>
      <p>{isLogin ? "Sign in to upload an MRI and run the AI pipeline." : "Create an account to access the MRI analysis workspace."}</p>
      {error && <div className="alert error">{error}</div>}
      <form onSubmit={submit} className="form-grid">
        {!isLogin && <>
          <Field label="Full name"><input value={form.name} onChange={(e) => update("name", e.target.value)} required /></Field>
          <Field label="Mobile"><input value={form.mobile} onChange={(e) => update("mobile", e.target.value)} inputMode="numeric" maxLength="10" required /></Field>
          <Field label="Age"><input type="number" value={form.age} onChange={(e) => update("age", e.target.value)} min="1" max="120" required /></Field>
          <Field label="Gender"><select value={form.gender} onChange={(e) => update("gender", e.target.value)}><option value="M">Male</option><option value="F">Female</option><option value="O">Other</option></select></Field>
        </>}
        <Field label="Email"><input type="email" value={form.email} onChange={(e) => update("email", e.target.value)} required /></Field>
        <Field label="Password"><input type="password" value={form.password} onChange={(e) => update("password", e.target.value)} required /></Field>
        {isLogin && <Field label="Account type"><select value={form.role} onChange={(e) => update("role", e.target.value)}><option value="user">User</option><option value="admin">Admin</option></select></Field>}
        {!isLogin && <Field label="Confirm password"><input type="password" value={form.confirm_password} onChange={(e) => update("confirm_password", e.target.value)} required /></Field>}
        <button className="button primary submit" disabled={busy}>{busy ? "Please wait…" : isLogin ? "Sign in →" : "Create account →"}</button>
      </form>
      <p className="auth-switch">{isLogin ? "Don't have an account?" : "Already have an account?"} <Link to={isLogin ? "/register" : "/login"}>{isLogin ? "Create one" : "Sign in"}</Link></p>
    </div>
  </section>;
}

function Field({ label, children }) {
  return <label className="field"><span>{label}</span>{children}</label>;
}

function ProtectedRoute({ user, adminOnly, children }) {
  const location = useLocation();
  if (!user) return <Navigate to="/login" state={{ from: location.pathname }} replace />;
  if (adminOnly && user.role !== "admin") return <Navigate to="/dashboard" replace />;
  return children;
}

function Dashboard({ user }) {
  const navigate = useNavigate();
  return <section className="page-section">
    <div className="dashboard-head">
      <div><span className="eyebrow">YOUR WORKSPACE</span><h1>Welcome, {user.name.split(" ")[0]}.</h1><p>Everything you need to run an MRI analysis.</p></div>
      <button className="button primary" onClick={() => navigate("/prediction")}>Analyze MRI →</button>
    </div>
    <div className="dashboard-grid">
      <div className="profile-card">
        <div className="profile-avatar">{user.name.charAt(0).toUpperCase()}</div>
        <h2>{user.name}</h2><p>{user.email}</p>
        <div className="profile-details"><span>Age <b>{user.age}</b></span><span>Gender <b>{user.gender}</b></span><span>Mobile <b>{user.mobile}</b></span></div>
      </div>
      <div className="workflow-card">
        <span className="eyebrow">QUICK START</span>
        <h2>Run the three-stage pipeline.</h2>
        <p>Upload a brain MRI. The backend will validate the image, classify tumour presence, and return a segmentation mask when applicable.</p>
        <button className="button ghost" onClick={() => navigate("/prediction")}>Open analyzer</button>
      </div>
    </div>
  </section>;
}

function Prediction() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const chooseFile = (next) => {
    setFile(next);
    setResult(null);
    setError("");
    if (next) setPreview(URL.createObjectURL(next));
    else setPreview("");
  };

  const analyze = async () => {
    if (!file) return;
    setBusy(true); setError(""); setResult(null);
    try {
      setResult(await api.predict(file));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const statusClass = useMemo(() => result?.status || "", [result]);

  return <section className="page-section analyzer">
    <div className="dashboard-head">
      <div><span className="eyebrow">AI ANALYSIS</span><h1>MRI Analyzer</h1><p>Upload one clear MRI image to run the complete inference pipeline.</p></div>
    </div>

    <div className="analyzer-grid">
      <div className="upload-card">
        <div className="dropzone">
          <input type="file" accept=".png,.jpg,.jpeg,.gif" onChange={(e) => chooseFile(e.target.files?.[0] || null)} />
          {preview ? <img src={preview} alt="MRI preview" className="preview-image" /> : <div className="upload-placeholder"><div className="upload-icon">＋</div><h3>Drop an MRI here</h3><p>PNG, JPG, JPEG or GIF · up to 12 MB</p><span className="button ghost compact">Choose image</span></div>}
        </div>
        {file && <div className="file-row"><span>{file.name}</span><button onClick={() => chooseFile(null)}>Remove</button></div>}
        <button className="button primary analyze-button" disabled={!file || busy} onClick={analyze}>
          {busy ? <><span className="spinner" /> Running AI pipeline…</> : "Run analysis →"}
        </button>
        {error && <div className="alert error">{error}</div>}
      </div>

      <div className="result-card">
        <span className="eyebrow">RESULT</span>
        {!result && !busy && <div className="empty-result"><div className="radar"><span /></div><h3>Awaiting an MRI</h3><p>Your classification and segmentation result will appear here.</p></div>}
        {busy && <div className="empty-result"><div className="loader-bars"><span /><span /><span /></div><h3>Analyzing image</h3><p>Running relevance detection, classification and segmentation as needed.</p></div>}
        {result && <div className={`result-content ${statusClass}`}>
          <div className="result-status"><span className="status-dot" /><strong>{result.message}</strong></div>
          {result.mask && <div className="mask-section"><div><span>Predicted region</span><img src={result.mask} alt="Predicted tumour segmentation mask" /></div></div>}
          <div className="result-note">AI output is an academic demonstration and should not be interpreted as a medical diagnosis.</div>
        </div>}
      </div>
    </div>
  </section>;
}

function AdminDashboard({ user }) {
  const [users, setUsers] = useState([]);
  const [search, setSearch] = useState("");
  const [message, setMessage] = useState("");
  const [passwords, setPasswords] = useState({ current_password: "", new_password: "", confirm_password: "" });

  const load = () => api.users().then((data) => setUsers(data.users)).catch((err) => setMessage(err.message));
  useEffect(() => { load(); }, []);

  const action = async (fn) => {
    setMessage("");
    try { const data = await fn(); setMessage(data.message); await load(); }
    catch (err) { setMessage(err.message); }
  };

  const savePassword = async (e) => {
    e.preventDefault();
    await action(() => api.changePassword(passwords));
    setPasswords({ current_password: "", new_password: "", confirm_password: "" });
  };

  const filtered = users.filter((item) =>
    [item.name, item.email, item.mobile].join(" ").toLowerCase().includes(search.toLowerCase())
  );

  return <section className="page-section">
    <div className="dashboard-head">
      <div><span className="eyebrow">ADMIN CONSOLE</span><h1>System overview.</h1><p>Manage registered users and account access.</p></div>
      <div className="admin-stat"><strong>{users.filter((u) => u.role === "user").length}</strong><span>registered users</span></div>
    </div>
    {message && <div className="alert success">{message}</div>}
    <div className="admin-layout">
      <div className="table-card">
        <div className="table-toolbar"><h2>Users</h2><input placeholder="Search users…" value={search} onChange={(e) => setSearch(e.target.value)} /></div>
        <div className="table-wrap"><table><thead><tr><th>Name</th><th>Email</th><th>Role</th><th>Status</th><th>Actions</th></tr></thead><tbody>
          {filtered.map((item) => <tr key={item.id}>
            <td><strong>{item.name}</strong><small>{item.mobile}</small></td><td>{item.email}</td><td><span className="tag">{item.role}</span></td><td><span className={`status-tag ${item.status}`}>{item.status}</span></td>
            <td className="actions">
              {item.status === "pending" && <button onClick={() => action(() => api.approve(item.id))}>Approve</button>}
              {item.status === "blocked" ? <button onClick={() => action(() => api.unblock(item.id))}>Unblock</button> : item.id !== user.id && <button onClick={() => action(() => api.block(item.id))}>Block</button>}
              {item.id !== user.id && <button className="danger-text" onClick={() => action(() => api.deleteUser(item.id))}>Delete</button>}
            </td>
          </tr>)}
          {!filtered.length && <tr><td colSpan="5" className="empty-cell">No users found.</td></tr>}
        </tbody></table></div>
      </div>
      <form className="password-card" onSubmit={savePassword}><span className="eyebrow">SECURITY</span><h2>Change admin password</h2><Field label="Current password"><input type="password" value={passwords.current_password} onChange={(e) => setPasswords({...passwords,current_password:e.target.value})} required /></Field><Field label="New password"><input type="password" value={passwords.new_password} onChange={(e) => setPasswords({...passwords,new_password:e.target.value})} required /></Field><Field label="Confirm new password"><input type="password" value={passwords.confirm_password} onChange={(e) => setPasswords({...passwords,confirm_password:e.target.value})} required /></Field><button className="button primary">Update password</button></form>
    </div>
  </section>;
}

function Footer() {
  return <footer className="footer"><div><strong>NeuroVision AI</strong><span>Brain MRI analysis research interface</span></div><span>Academic project · Not a medical diagnosis</span></footer>;
}

function Splash() {
  return <div className="splash"><div className="brand-mark large"><span /></div><div className="loader-bars"><span /><span /><span /></div></div>;
}

export default App;