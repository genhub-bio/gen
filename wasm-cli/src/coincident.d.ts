// coincident (https://www.npmjs.com/package/coincident) ships no published type declarations for
// the subpaths used here.
declare module 'coincident/main' {
  export default function coincident(): { Worker: typeof Worker };
}

declare module 'coincident/worker' {
  export default function coincident(): Promise<{ proxy: any }>;
}
