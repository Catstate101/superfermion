import type { BaseLayoutProps } from "fumadocs-ui/layouts/shared";

export const baseOptions: BaseLayoutProps = {
  nav: {
    title: (
      <span className="flex items-center gap-2 font-semibold">
        <span className="flex h-6 w-6 items-center justify-center bg-violet-600/10 font-mono text-xs font-bold text-violet-600">
          sf
        </span>
        Superfermion docs
      </span>
    ),
  },
  links: [
    { text: "GitHub", url: "https://github.com/Catstate101/superfermion" },
    { text: "PyPI", url: "https://pypi.org/project/superfermion" },
  ],
};
