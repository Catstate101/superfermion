import { createMDX } from "fumadocs-mdx/next";

const withMDX = createMDX();

const config = {
  output: "export" as const,
  images: { unoptimized: true },
};

export default withMDX(config);
