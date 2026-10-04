const API_URL = (import.meta.env.VITE_API_URL || "https://brain-tumour-ai-backend-onnx.onrender.com").replace(/\/$/, "");

async function request(path, options = {}) {
  const response = await fetch(`${API_URL}${path}`, {
    credentials: "include",
    ...options,
    headers: {
      ...(options.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...(options.headers || {}),
    },
  });

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(data.message || "Something went wrong.");
  }

  return data;
}

export const api = {
  me: () => request("/api/auth/me"),
  login: (payload) => request("/api/auth/login", {
    method: "POST",
    body: JSON.stringify(payload),
  }),
  register: (payload) => request("/api/auth/register", {
    method: "POST",
    body: JSON.stringify(payload),
  }),
  logout: () => request("/api/auth/logout", { method: "POST" }),
  predict: (file) => {
    const form = new FormData();
    form.append("image", file);
    return request("/api/predict", { method: "POST", body: form });
  },
  users: () => request("/api/admin/users"),
  block: (id) => request(`/api/admin/users/${id}/block`, { method: "POST" }),
  unblock: (id) => request(`/api/admin/users/${id}/unblock`, { method: "POST" }),
  approve: (id) => request(`/api/admin/users/${id}/approve`, { method: "POST" }),
  deleteUser: (id) => request(`/api/admin/users/${id}`, { method: "DELETE" }),
  changePassword: (payload) => request("/api/admin/profile/password", {
    method: "POST",
    body: JSON.stringify(payload),
  }),
};