"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";

const stats = [
  { label: "注册犬只", value: "1,200+", icon: "🐾" },
  { label: "识别准确率", value: "87.9%", icon: "🎯" },
  { label: "平均响应时间", value: "<2s", icon: "⚡" },
  { label: "数据安全", value: "100%", icon: "🔒" },
];

const features = [
  {
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
          d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
      </svg>
    ),
    title: "孪生神经网络",
    desc: "采用 ResNet-50 为骨干的孪生网络架构，通过对比学习实现高精度犬鼻纹匹配",
  },
  {
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
          d="M4 7v10c0 2 1 3 3 3h10c2 0 3-1 3-3V7c0-2-1-3-3-3H7C5 4 4 5 4 7zm5 3h6M9 14h4" />
      </svg>
    ),
    title: "云端数据库",
    desc: "基于 Supabase 的安全云存储，所有犬鼻纹图像和档案信息均加密存储",
  },
  {
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
          d="M13 10V3L4 14h7v7l9-11h-7z" />
      </svg>
    ),
    title: "实时比对",
    desc: "上传图片即时与数据库所有已注册犬只进行比对，秒级返回识别结果",
  },
];

export default function Home() {
  const router = useRouter();

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      {/* ── Navigation ─────────────────────────────────────────────── */}
      <nav className="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-gray-100 shadow-sm">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
                <span className="text-white text-lg">🐾</span>
              </div>
              <span className="font-bold text-gray-900 text-lg">犬鼻纹识别系统</span>
            </div>
            <div className="flex items-center gap-2">
              <Link href="/search" className="nav-link">
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
                查询
              </Link>
              <Link href="/register" className="btn-primary text-sm px-4 py-2">
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M12 4v16m8-8H4" />
                </svg>
                登记
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* ── Hero Section ───────────────────────────────────────────── */}
      <section className="relative overflow-hidden">
        <div className="hero-gradient">
          <div className="absolute inset-0 opacity-10">
            <div className="absolute top-0 left-0 w-96 h-96 bg-white rounded-full -translate-x-1/2 -translate-y-1/2" />
            <div className="absolute bottom-0 right-0 w-96 h-96 bg-white rounded-full translate-x-1/3 translate-y-1/3" />
          </div>
          <div className="relative max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-20 text-center">
            <div className="inline-flex items-center gap-2 px-4 py-1.5 bg-white/20 rounded-full text-white text-sm font-medium mb-6 backdrop-blur-sm border border-white/30">
              <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
              系统运行正常
            </div>
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold text-white mb-6 leading-tight">
              犬鼻纹智能识别平台
            </h1>
            <p className="text-xl text-blue-100 mb-10 max-w-2xl mx-auto leading-relaxed">
              每只狗的鼻纹都是独一无二的「指纹」。
              借助孪生神经网络，我们让犬只身份识别触手可及。
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button
                onClick={() => router.push("/search")}
                className="inline-flex items-center justify-center gap-2 px-8 py-4 rounded-xl font-bold
                           text-blue-700 bg-white hover:bg-blue-50
                           transition-all duration-200 shadow-lg hover:shadow-xl text-base"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
                立即查询犬只
              </button>
              <button
                onClick={() => router.push("/register")}
                className="inline-flex items-center justify-center gap-2 px-8 py-4 rounded-xl font-bold
                           text-white bg-white/20 hover:bg-white/30 border border-white/40
                           transition-all duration-200 backdrop-blur-sm text-base"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M12 4v16m8-8H4" />
                </svg>
                登记新犬只
              </button>
            </div>
          </div>
        </div>

        {/* Stats Bar */}
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="bg-white rounded-2xl shadow-lg -mt-8 p-6 grid grid-cols-2 md:grid-cols-4 gap-6">
            {stats.map((s, i) => (
              <div key={i} className="text-center">
                <div className="text-2xl mb-1">{s.icon}</div>
                <div className="text-2xl font-bold text-gray-900">{s.value}</div>
                <div className="text-sm text-gray-500 mt-0.5">{s.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Main Cards ─────────────────────────────────────────────── */}
      <section className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-3">选择功能</h2>
          <p className="text-gray-500 text-lg">两步操作，轻松管理您的爱犬档案</p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Search Card */}
          <div
            onClick={() => router.push("/search")}
            className="group relative bg-white rounded-2xl p-8 border border-gray-100 shadow-sm
                       hover:shadow-xl hover:border-blue-200 transition-all duration-300 cursor-pointer overflow-hidden"
          >
            <div className="absolute top-0 right-0 w-40 h-40 bg-blue-50 rounded-full -translate-y-1/2 translate-x-1/2 group-hover:scale-150 transition-transform duration-500" />
            <div className="relative">
              <div className="w-16 h-16 bg-blue-100 rounded-2xl flex items-center justify-center mb-6
                              group-hover:bg-blue-600 transition-colors duration-300">
                <svg className="w-8 h-8 text-blue-600 group-hover:text-white transition-colors duration-300"
                     fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                    d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </div>
              <h3 className="text-2xl font-bold text-gray-900 mb-3">查询犬类信息</h3>
              <p className="text-gray-500 mb-6 leading-relaxed">
                上传犬鼻纹照片，系统自动与数据库中所有已登记犬只进行智能比对，
                快速找到匹配档案。
              </p>
              <div className="flex items-center gap-2 text-blue-600 font-semibold group-hover:gap-3 transition-all duration-200">
                立即查询
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </div>
            </div>
          </div>

          {/* Register Card */}
          <div
            onClick={() => router.push("/register")}
            className="group relative bg-white rounded-2xl p-8 border border-gray-100 shadow-sm
                       hover:shadow-xl hover:border-emerald-200 transition-all duration-300 cursor-pointer overflow-hidden"
          >
            <div className="absolute top-0 right-0 w-40 h-40 bg-emerald-50 rounded-full -translate-y-1/2 translate-x-1/2 group-hover:scale-150 transition-transform duration-500" />
            <div className="relative">
              <div className="w-16 h-16 bg-emerald-100 rounded-2xl flex items-center justify-center mb-6
                              group-hover:bg-emerald-500 transition-colors duration-300">
                <svg className="w-8 h-8 text-emerald-600 group-hover:text-white transition-colors duration-300"
                     fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                    d="M12 4v16m8-8H4" />
                </svg>
              </div>
              <h3 className="text-2xl font-bold text-gray-900 mb-3">登记新犬只</h3>
              <p className="text-gray-500 mb-6 leading-relaxed">
                为您的爱犬建立专属电子档案，上传鼻纹照片并填写基本信息，
                系统将自动检测是否已存在记录。
              </p>
              <div className="flex items-center gap-2 text-emerald-600 font-semibold group-hover:gap-3 transition-all duration-200">
                开始登记
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── How it works ───────────────────────────────────────────── */}
      <section className="bg-white border-y border-gray-100">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-3">工作原理</h2>
            <p className="text-gray-500">三步完成犬只身份识别</p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              { step: "01", title: "上传照片", desc: "拍摄清晰的犬鼻正面照片上传至系统，支持 JPG/PNG 格式" },
              { step: "02", title: "AI 比对", desc: "孪生网络提取特征向量，与数据库中所有记录逐一进行余弦相似度计算" },
              { step: "03", title: "获取结果", desc: "实时返回匹配结果和置信度分数，展示最佳匹配的犬只档案信息" },
            ].map((item, i) => (
              <div key={i} className="relative text-center">
                {i < 2 && (
                  <div className="hidden md:block absolute top-8 left-full w-full h-0.5 bg-gradient-to-r from-blue-200 to-transparent z-0" />
                )}
                <div className="relative z-10">
                  <div className="w-16 h-16 bg-blue-600 rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-lg shadow-blue-200">
                    <span className="text-white font-bold text-lg">{item.step}</span>
                  </div>
                  <h3 className="text-lg font-bold text-gray-900 mb-2">{item.title}</h3>
                  <p className="text-gray-500 text-sm leading-relaxed">{item.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Features ───────────────────────────────────────────────── */}
      <section className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-3">技术特性</h2>
          <p className="text-gray-500">基于前沿深度学习技术构建</p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {features.map((f, i) => (
            <div key={i} className="bg-white rounded-2xl p-6 border border-gray-100 shadow-sm hover:shadow-md transition-shadow duration-200">
              <div className="w-12 h-12 bg-blue-50 rounded-xl flex items-center justify-center text-blue-600 mb-4">
                {f.icon}
              </div>
              <h3 className="font-bold text-gray-900 mb-2">{f.title}</h3>
              <p className="text-gray-500 text-sm leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── Footer ─────────────────────────────────────────────────── */}
      <footer className="border-t border-gray-100 bg-white mt-8">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-2 text-gray-600">
              <span className="text-xl">🐾</span>
              <span className="font-semibold">犬鼻纹识别系统</span>
            </div>
            <p className="text-sm text-gray-400">
              基于孪生神经网络（Siamese Network + ResNet-50）的细粒度生物特征识别系统
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
