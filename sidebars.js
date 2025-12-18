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
    {
      type: 'category',
      label: '📚 Modul 5: RAG',
      link: {
        type: 'generated-index',
        title: 'Retrieval Augmented Generation',
        description: 'Menghubungkan LLM dengan knowledge base eksternal.',
      },
      items: [
        'rag/konsep-rag',
        'rag/document-loaders',
        'rag/text-splitters',
        'rag/embeddings',
        'rag/vector-stores',
        'rag/retrievers',
        'rag/basic-rag-chain',
        'rag/advanced-retrieval',
        'rag/rag-evaluation',
        'rag/rag-best-practices',
      ],
    },
    {
      type: 'category',
      label: '🧠 Modul 6: Memory',
      link: {
        type: 'generated-index',
        title: 'Memory & Conversation',
        description: 'Membuat chatbot yang mengingat konteks percakapan.',
      },
      items: [
        'memory/konsep-memory',
        'memory/message-history',
        'memory/runnable-with-history',
        'memory/memory-types',
        'memory/conversational-rag',
      ],
    },
    {
      type: 'category',
<<<<<<< HEAD
      label: '🤖 Modul 7: Agents',
      link: {
        type: 'generated-index',
        title: 'Agents & Tool Calling',
        description: 'Membiarkan LLM mengambil aksi dan menggunakan tools.',
      },
      items: [
        'agents/konsep-agents',
        'agents/tool-calling',
        'agents/built-in-tools',
        'agents/custom-tools',
        'agents/agent-executors',
        'agents/agent-patterns',
        'agents/streaming-agents',
      ],
    },
    {
      type: 'category',
      label: '📊 Modul 8: LangGraph',
      link: {
        type: 'generated-index',
        title: 'LangGraph',
        description: 'Membangun aplikasi AI kompleks dengan state management.',
      },
      items: [
        'langgraph/intro',
        'langgraph/core-concepts',
        'langgraph/conditional-edges',
        'langgraph/checkpointing',
        'langgraph/human-in-loop',
        'langgraph/multi-agent',
        'langgraph/subgraphs',
        'langgraph/studio',
      ],
    },
    {
      type: 'category',
      label: '🚀 Modul 9: Production',
      link: {
        type: 'generated-index',
        title: 'Production & Observability',
        description: 'Deploy dan monitor aplikasi LangChain di production.',
      },
      items: [
        'production/langsmith-setup',
        'production/tracing',
        'production/evaluation',
        'production/langserve',
        'production/best-practices',
      ],
    },
    {
      type: 'category',
      label: '🎯 Modul 10: Proyek Praktis',
      link: {
        type: 'generated-index',
        title: 'Proyek Praktis',
        description: 'Mengaplikasikan semua yang dipelajari dalam proyek nyata.',
      },
      items: [
        'projects/overview',
        'projects/rag-chatbot',
        'projects/research-agent',
        'projects/customer-support',
        'projects/content-pipeline',
      ],
    },
  ],
};

export default sidebars;



