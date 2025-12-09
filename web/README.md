# Torah Source Finder - Web Interface

React/Next.js web interface for managing and searching Torah sources from Sefaria.

## Features

- Browse books organized by categories
- Run pipeline processing for individual books
- Search within specific books or across all books
- View search results with scores and references

## Setup

1. Install dependencies:
```bash
npm install
# or
yarn install
# or
pnpm install
```

2. Make sure the API server is running on `http://localhost:8000`

3. Run the development server:
```bash
npm run dev
# or
yarn dev
# or
pnpm dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser

## Environment Variables

Create a `.env.local` file if you need to change the API URL:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## API Endpoints Used

- `GET /api/books` - Get all books organized by categories
- `POST /api/pipeline` - Run pipeline for books
- `POST /api/search` - Search for text in books



























