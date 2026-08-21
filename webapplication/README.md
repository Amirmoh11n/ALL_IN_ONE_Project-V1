# webapplication

The user-facing system: a FastAPI backend and a React frontend that let a user upload an MRI image and receive
a text prediction (one of the 4 tumor classes).

- `backend/` — FastAPI service exposing the prediction endpoint, calling into `src/inference`.
- `frontend/` — React application (upload UI + result display).
