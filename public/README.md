# Public Folder

This folder contains publicly accessible files that can be accessed by anyone without authentication.

## Usage

Any file placed in this folder will be accessible via the `/public/` URL path.

### Example

If you place a file named `example.pdf` in this folder, it will be accessible at:
```
http://your-server/public/example.pdf
```

## Use Cases

- Public documentation files
- Images for public display
- Downloadable resources
- Any static content that should be accessible to all users

## Security Note

Do not place sensitive or private files in this folder as they will be publicly accessible to anyone with the URL.
