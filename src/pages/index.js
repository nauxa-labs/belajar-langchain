import React from 'react';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import styles from './index.module.css';

function HeroBanner() {
  return (
    <header className={styles.heroBanner}>
      <div className="container">
        <h1 className={styles.heroTitle}>📚 Belajar LangChain</h1>
        <p className={styles.heroSubtitle}>
          Panduan lengkap belajar LangChain dari nol dalam Bahasa Indonesia
        </p>
        <p className={styles.heroDescription}>
          Ingin membangun aplikasi AI seperti chatbot, asisten pintar, atau sistem RAG?
          Kamu berada di tempat yang tepat!
        </p>
        <div className={styles.buttons}>
          <Link
            className="button button--primary button--lg"
            to="/docs/prasyarat/pengantar-genai">
            🚀 Mulai Belajar
          </Link>
          <Link
            className="button button--secondary button--lg"
            to="https://github.com/nauxa-labs/belajar-langchain">
            ⭐ GitHub
          </Link>
        </div>
      </div>
    </header>
  );
}

const curriculum = [
  {
    status: '✅',
    title: 'Prasyarat & Setup',
    description: 'Setup environment Python, API keys, pengantar AI',
    link: '/docs/prasyarat/pengantar-genai',
  },
  {
    status: '✅',
    title: 'Fondasi LangChain',
    description: 'Chat models, prompt templates, output parsers',
    link: '/docs/fondasi/chat-models-vs-llms',
  },
  {
    status: '✅',
    title: 'LCEL',
    description: 'Expression language untuk membangun chains',
    link: '/docs/lcel/filosofi-lcel',
  },
  {
    status: '✅',
    title: 'Prompt Engineering',
    description: 'Menulis prompt efektif, few-shot, debugging',
    link: '/docs/prompt-engineering/prinsip-prompting',
  },
  {
    status: '✅',
    title: 'Structured Output',
    description: 'Pydantic, parsing JSON, typed responses',
    link: '/docs/structured-output/mengapa-structured-output',
  },
  {
    status: '⏳',
    title: 'RAG',
    description: 'Retrieval Augmented Generation',
    link: null,
  },
  {
    status: '⏳',
    title: 'Tools & Function Calling',
    description: 'Koneksi ke external APIs',
    link: null,
  },
  {
    status: '⏳',
    title: 'Memory & State',
    description: 'Chatbot dengan memori percakapan',
    link: null,
  },
  {
    status: '⏳',
    title: 'Agents',
    description: 'AI yang bisa mengambil keputusan',
    link: null,
  },
  {
    status: '⏳',
    title: 'LangGraph',
    description: 'Multi-agent workflows',
    link: null,
  },
  {
    status: '⏳',
    title: 'Production',
    description: 'Deployment & monitoring',
    link: null,
  },
];

function CurriculumSection() {
  return (
    <section className={styles.curriculum}>
      <div className="container">
        <h2>✨ Apa yang Akan Kamu Pelajari?</h2>
        <div className={styles.curriculumGrid}>
          {curriculum.map((item, idx) => (
            <div key={idx} className={`${styles.curriculumCard} ${item.status === '⏳' ? styles.comingSoon : ''}`}>
              <span className={styles.status}>{item.status}</span>
              <h3>
                {item.link ? (
                  <Link to={item.link}>{item.title}</Link>
                ) : (
                  item.title
                )}
              </h3>
              <p>{item.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

const audiences = [
  { emoji: '👨‍💻', title: 'Developer', desc: 'yang ingin menambah skill AI/LLM' },
  { emoji: '🎓', title: 'Mahasiswa', desc: 'yang mempelajari AI aplikatif' },
  { emoji: '🚀', title: 'Startup Founder', desc: 'yang ingin membangun produk AI' },
  { emoji: '🔄', title: 'Career Switcher', desc: 'ke bidang AI engineering' },
];

function AudienceSection() {
  return (
    <section className={styles.audience}>
      <div className="container">
        <h2>🎯 Untuk Siapa?</h2>
        <div className={styles.audienceGrid}>
          {audiences.map((item, idx) => (
            <div key={idx} className={styles.audienceCard}>
              <span className={styles.emoji}>{item.emoji}</span>
              <h3>{item.title}</h3>
              <p>{item.desc}</p>
            </div>
          ))}
        </div>
        <p className={styles.prereq}>
          <strong>Prasyarat:</strong> Familiar dengan Python dasar sudah cukup!
        </p>
      </div>
    </section>
  );
}

const whyLangchain = [
  { icon: '⚡', title: 'Abstraksi Mudah', desc: 'Tidak perlu handle low-level API' },
  { icon: '🔌', title: 'Multi-Provider', desc: 'OpenAI, Anthropic, Google dalam satu interface' },
  { icon: '🧩', title: 'Composable', desc: 'Bangun sistem kompleks dari komponen sederhana' },
  { icon: '📦', title: 'Batteries Included', desc: 'RAG, agents, memory siap pakai' },
];

function WhySection() {
  return (
    <section className={styles.why}>
      <div className="container">
        <h2>💡 Kenapa LangChain?</h2>
        <div className={styles.whyGrid}>
          {whyLangchain.map((item, idx) => (
            <div key={idx} className={styles.whyCard}>
              <span className={styles.icon}>{item.icon}</span>
              <h3>{item.title}</h3>
              <p>{item.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function Footer() {
  return (
    <section className={styles.footerCta}>
      <div className="container">
        <h2>🏃 Siap Mulai?</h2>
        <p>Mulai dari Modul 0 dan ikuti setiap bab secara berurutan.</p>
        <Link
          className="button button--primary button--lg"
          to="/docs/prasyarat/pengantar-genai">
          Mulai Belajar Sekarang →
        </Link>
        <p className={styles.footerNote}>
          Disusun oleh <a href="https://github.com/nauxa-labs">Nauxa Labs</a> · Assisted with AI 🤖
        </p>
      </div>
    </section>
  );
}

export default function Home() {
  return (
    <Layout
      title="Belajar LangChain"
      description="Panduan lengkap belajar LangChain dari nol dalam Bahasa Indonesia">
      <HeroBanner />
      <main>
        <CurriculumSection />
        <AudienceSection />
        <WhySection />
        <Footer />
      </main>
    </Layout>
  );
}
