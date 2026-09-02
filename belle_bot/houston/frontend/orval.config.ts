import { defineConfig } from 'orval';

export default defineConfig({
  houston: {
    input: '../openapi.json',
    output: {
      target: 'src/api/api.ts',
      client: 'fetch',
      baseUrl: 'http://localhost:8085',
      override: {
        mutator: {
          path: './src/api/mutator.ts',
          name: 'customInstance',
        },
      },
    },
  },
});
