// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).

import { themes as prismThemes } from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Belajar LangChain',
  tagline: 'Panduan komprehensif belajar LangChain dari nol dalam Bahasa Indonesia',
  favicon: 'img/favicon.ico',

  // Production URL
  url: 'https://nauxa-labs.github.io',
  baseUrl: '/',

  // GitHub pages deployment config
  organizationName: 'nauxa-labs',
  projectName: 'belajar-langchain',

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',

  // Localization
  i18n: {
    defaultLocale: 'id',
    locales: ['id'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          // Edit this page link
          editUrl: 'https://github.com/nauxa-labs/belajar-langchain/tree/main/',
          // Disable last update time (requires git repository)
          showLastUpdateTime: false,
        },
        blog: false, // Disable blog
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Social card
      image: 'img/social-card.jpg',

      navbar: {
        title: 'Belajar LangChain',
        logo: {
          alt: 'Belajar LangChain Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Dokumentasi',
          },
          {
            href: 'https://github.com/nauxa-labs/belajar-langchain',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },

      footer: {
        style: 'dark',
        links: [
          {
            title: 'Dokumentasi',
            items: [
              {
                label: 'Mulai Belajar',
                to: '/docs/prasyarat/pengantar-genai',
              },
            ],
          },
          {
            title: 'Komunitas',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/nauxa-labs/belajar-langchain',
              },
            ],
          },
          {
            title: 'Sumber Daya',
            items: [
              {
                label: 'LangChain Docs',
                href: 'https://python.langchain.com/',
              },
              {
                label: 'LangChain GitHub',
                href: 'https://github.com/langchain-ai/langchain',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Belajar LangChain. Built with Docusaurus.`,
      },

      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: ['python', 'bash', 'json'],
      },

      // Table of contents
      tableOfContents: {
        minHeadingLevel: 2,
        maxHeadingLevel: 4,
      },

      // Announcement bar (optional)
      announcementBar: {
        id: 'support_us',
        content: '🚀 <b>Belajar LangChain</b> - Panduan komprehensif LangChain dalam Bahasa Indonesia!',
        backgroundColor: '#4CAF50',
        textColor: '#ffffff',
        isCloseable: true,
      },
    }),
};

export default config;
