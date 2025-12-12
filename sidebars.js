// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'category',
      label: '🚀 Modul 0: Prasyarat',
      link: {
        type: 'generated-index',
        title: 'Prasyarat & Lingkungan Setup',
        description: 'Memastikan environment siap dan memahami landscape AI/LLM.',
      },
      items: [
        'prasyarat/pengantar-genai',
        'prasyarat/mengapa-langchain',
        'prasyarat/setup-environment',
        'prasyarat/manajemen-api-keys',
      ],
    },
    {
      type: 'category',
      label: '🧱 Modul 1: Fondasi',
      link: {
        type: 'generated-index',
        title: 'Fondasi LangChain',
        description: 'Memahami building blocks dasar LangChain.',
      },
      items: [
        'fondasi/chat-models-vs-llms',
        'fondasi/memanggil-model',
        'fondasi/prompt-templates',
        'fondasi/output-parsers',
        'fondasi/menggabungkan-komponen',
      ],
    },
    {
      type: 'category',
      label: '⛓️ Modul 2: LCEL',
      link: {
        type: 'generated-index',
        title: 'LangChain Expression Language',
        description: 'Cara modern membangun chains di LangChain.',
      },
      items: [
        'lcel/filosofi-lcel',
        'lcel/runnable-interface',
        'lcel/composing-runnables',
        'lcel/branching-routing',
        'lcel/error-handling',
        'lcel/streaming',
      ],
    },
    {
      type: 'category',
      label: '✍️ Modul 3: Prompt Engineering',
      link: {
        type: 'generated-index',
        title: 'Prompt Engineering',
        description: 'Menulis prompt yang efektif dan maintainable.',
      },
      items: [
        'prompt-engineering/prinsip-prompting',
        'prompt-engineering/few-shot-prompting',
        'prompt-engineering/advanced-techniques',
        'prompt-engineering/langchain-hub',
        'prompt-engineering/debugging-prompts',
      ],
    },
    {
      type: 'category',
      label: '📦 Modul 4: Structured Output',
      link: {
        type: 'generated-index',
        title: 'Structured Output',
        description: 'Memaksa LLM menghasilkan data terstruktur.',
      },
      items: [
        'structured-output/mengapa-structured-output',
        'structured-output/pydantic-models',
        'structured-output/with-structured-output',
        'structured-output/output-parsers-advanced',
      ],
    },
    // Modul 5-10 akan ditambahkan saat kontennya dibuat
  ],
};

export default sidebars;
