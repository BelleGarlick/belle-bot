import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { FabricContextProvider } from './contexts/ServerContext.tsx'

createRoot(document.getElementById('root')!).render(
    <StrictMode>
        <FabricContextProvider>
            <App />
        </FabricContextProvider>
    </StrictMode>,
)
