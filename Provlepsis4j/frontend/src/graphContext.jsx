import { createContext, useContext, useState } from 'react';

const Ctx = createContext({ graphName: null, setGraphName: () => {} });

export function GraphProvider({ children }) {
  const [graphName, setGraphName] = useState(null);
  return <Ctx.Provider value={{ graphName, setGraphName }}>{children}</Ctx.Provider>;
}

export function useGraph() {
  return useContext(Ctx);
}
