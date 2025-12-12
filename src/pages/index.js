import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import styles from './index.module.css';

function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <h1 className="hero__title">{siteConfig.title}</h1>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/prasyarat/pengantar-genai">
            🚀 Mulai Belajar
          </Link>
        </div>
      </div>
    </header>
  );
}

function Feature({ emoji, title, description }) {
  return (
    <div className={clsx('col col--4', styles.feature)}>
      <div className="text--center">
        <span style={{ fontSize: '4rem' }}>{emoji}</span>
      </div>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

const FeatureList = [
  {
    emoji: '📚',
    title: 'Komprehensif',
    description: '10 modul dengan 60+ bab yang mencakup semua aspek LangChain, dari dasar hingga production.',
  },
  {
    emoji: '🇮🇩',
    title: 'Bahasa Indonesia',
    description: 'Dokumentasi lengkap dalam Bahasa Indonesia dengan istilah teknis yang jelas.',
  },
  {
    emoji: '💻',
    title: 'Hands-On',
    description: 'Setiap bab dilengkapi dengan code examples yang bisa langsung dijalankan.',
  },
  {
    emoji: '🎯',
    title: 'Use Case Praktis',
    description: 'Setiap modul diakhiri dengan proyek praktis untuk mengaplikasikan pembelajaran.',
  },
  {
    emoji: '🔄',
    title: 'Selalu Update',
    description: 'Mengikuti perkembangan LangChain terbaru dengan best practices terkini.',
  },
  {
    emoji: '🤖',
    title: 'Production Ready',
    description: 'Pelajari cara deploy, monitor, dan scale aplikasi LangChain di production.',
  },
];

function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <h2 className="text--center margin-bottom--lg">Kenapa Belajar di Sini?</h2>
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}

function ModuleOverview() {
  const modules = [
    { num: '0', title: 'Prasyarat & Setup', desc: 'Environment & API Keys' },
    { num: '1', title: 'Fondasi', desc: 'Models, Prompts, Parsers' },
    { num: '2', title: 'LCEL', desc: 'Expression Language' },
    { num: '3', title: 'Prompt Engineering', desc: 'Few-shot, CoT, Hub' },
    { num: '4', title: 'Structured Output', desc: 'Pydantic & Validation' },
    { num: '5', title: 'RAG', desc: 'Retrieval Augmented Gen' },
    { num: '6', title: 'Memory', desc: 'Conversation History' },
    { num: '7', title: 'Agents', desc: 'Tool Calling & Actions' },
    { num: '8', title: 'LangGraph', desc: 'State Machines' },
    { num: '9', title: 'Production', desc: 'Deploy & Monitor' },
    { num: '10', title: 'Proyek', desc: '4 Real-World Projects' },
  ];

  return (
    <section className={styles.modules}>
      <div className="container">
        <h2 className="text--center margin-bottom--lg">📖 Struktur Kurikulum</h2>
        <div className={styles.moduleGrid}>
          {modules.map((m, idx) => (
            <div key={idx} className={styles.moduleCard}>
              <div className={styles.moduleNum}>{m.num}</div>
              <div className={styles.moduleInfo}>
                <strong>{m.title}</strong>
                <small>{m.desc}</small>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title} - Panduan LangChain Bahasa Indonesia`}
      description="Panduan komprehensif belajar LangChain dari nol dalam Bahasa Indonesia. Mencakup RAG, Agents, LangGraph, dan production deployment.">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
        <ModuleOverview />
      </main>
    </Layout>
  );
}
