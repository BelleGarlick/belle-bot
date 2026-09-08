# Houston Frontend

The Houston frontend is a React application built with TypeScript and Vite. It provides a visual interface for managing and replaying bot events.

## Setup and Development

### 1. Install Dependencies
Navigate to the frontend directory and install the necessary npm packages:

```bash
cd houston/frontend
npm install
```

### 2. Development Mode
To start the development server with Hot Module Replacement (HMR):

```bash
cd houston/frontend
npm run dev
```

### 3. Build for Production
To build the application for production, you can use the script in the `scripts` directory:

```bash
./houston/scripts/build-fe.sh
```

The output will be saved in `houston/frontend/dist`.

## API Client Generation

The frontend uses an automatically generated TypeScript client to interact with the Houston API. If the backend API changes, you need to regenerate the client.

### Using the build script:
The `build.sh` script automates the process of exporting the OpenAPI schema from the FastAPI server and generating the TypeScript types using `orval`.

```bash
./houston/scripts/build.sh
```

### Manual regeneration:
1. Ensure the server's OpenAPI schema is exported to `houston/openapi.json`.
2. Run orval in the frontend directory:
   ```bash
   cd houston/frontend
   npx orval --config orval.config.ts
   ```

## Technologies Used
- **React**: UI library.
- **TypeScript**: Static typing.
- **Vite**: Build tool and dev server.
- **Orval**: API client generator.
