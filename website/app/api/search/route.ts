import { createFromSource } from "fumadocs-core/search/server";
import { source } from "@/lib/source";

// `output: "export"` cannot serve request-time handlers, so the index is
// prerendered once at build time into `out/api/search` and downloaded by the
// browser (`search.options.type = "static"` in the root provider), which then
// runs the Orama query locally.
export const revalidate = false;

export const { staticGET: GET } = createFromSource(source);
