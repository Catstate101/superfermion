import type { BaseLayoutProps } from "fumadocs-ui/layouts/shared";

export const baseOptions: BaseLayoutProps = {
  nav: {
    title: (
      <span className="group flex items-center gap-2 font-semibold">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src="/logo.png"
          alt="Superfermion logo"
          title="Superfermion"
          className="h-10 w-auto origin-left rounded-md transition-transform duration-300 ease-out group-hover:scale-150 md:h-12"
        />
        <span className="hidden sm:inline">Superfermion docs</span>
      </span>
    ),
  },
  links: [
    {
      text: "GitHub",
      url: "https://github.com/Catstate101/superfermion",
      external: true,
    },
    {
      text: "Pull requests",
      url: "https://github.com/Catstate101/superfermion/pulls",
      external: true,
    },
    {
      text: "PyPI",
      url: "https://pypi.org/project/superfermion",
      external: true,
    },
    {
      text: "Paper (DOI)",
      url: "https://doi.org/10.5281/zenodo.23574613",
      external: true,
    },
  ],
};
