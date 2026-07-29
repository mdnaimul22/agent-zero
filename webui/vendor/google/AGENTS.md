# Google Icons Vendor DOX

## Purpose

- Own vendored Google Material Symbols font and stylesheet assets.

## Ownership

- `google-icons.css` owns the font-face and icon class definitions.
- `google-icons.woff2` is the primary compressed font binary; `google-icons.ttf` is its compatibility fallback.

## Local Contracts

- Keep the WOFF2 and TTF fonts equivalent and keep their CSS references synchronized.
- Icon ligature elements must remain exactly one em wide and high, clip fallback text, and stay transparent until the Font Loading API confirms the Material Symbols face. A failed font may leave a blank square but must never expose a layout-breaking ligature name.
- Do not add unrelated remote font assets or tracking references.

## Work Guidance

- Replace both font formats and the stylesheet together when updating icon coverage.

## Verification

- Smoke-test visible icon surfaces after changes.

## Child DOX Index

No child DOX files.
