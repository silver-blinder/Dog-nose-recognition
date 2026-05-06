"use client";

import axios from "axios";
import Link from "next/link";
import { useCallback, useState } from "react";

function NavBar() {
  return (
    <nav className="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-gray-100 shadow-sm">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 py-4 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2 text-gray-700 hover:text-blue-600 transition-colors">
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          <span className="font-medium">返回首页</span>
        </Link>
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 bg-blue-600 rounded-md flex items-center justify-center">
            <span className="text-white text-xs">🐾</span>
          </div>
          <span className="font-semibold text-gray-800 hidden sm:block">犬鼻纹识别系统</span>
        </div>
      </div>
    </nav>
  );
}

function UploadZone({
  preview,
  onFile,
  onReset,
}: {
  preview: string;
  onFile: (f: File) => void;
  onReset: () => void;
}) {
  const [dragging, setDragging] = useState(false);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith("image/")) onFile(file);
    },
    [onFile]
  );

  return (
    <div
      className={`relative w-full h-72 rounded-2xl border-2 border-dashed transition-all duration-300 overflow-hidden
        ${dragging ? "border-blue-500 bg-blue-50 scale-[1.01]" : "border-gray-200 bg-gray-50 hover:border-blue-300 hover:bg-blue-50/50"}
        ${preview ? "border-solid border-blue-200 bg-white" : "cursor-pointer"}`}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
    >
      {preview ? (
        <>
          <img src={preview} alt="预览" className="w-full h-full object-contain p-4" />
          <button
            onClick={onReset}
            className="absolute top-3 right-3 w-8 h-8 bg-gray-900/60 hover:bg-gray-900/80 rounded-full
                       flex items-center justify-center text-white text-lg transition-colors"
          >
            ×
          </button>
          <div className="absolute bottom-3 left-1/2 -translate-x-1/2">
            <span className="badge-info text-xs">图片已选择</span>
          </div>
        </>
      ) : (
        <label htmlFor="noseImage" className="flex flex-col items-center justify-center h-full cursor-pointer p-8 text-center">
          <div className="w-16 h-16 bg-blue-100 rounded-2xl flex items-center justify-center mb-4">
            <svg className="w-8 h-8 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
          </div>
          <p className="text-gray-700 font-semibold text-lg mb-1">点击上传或拖拽图片至此</p>
          <p className="text-gray-400 text-sm">支持 JPG、PNG 格式 · 建议使用正面清晰照片</p>
          <input
            type="file"
            id="noseImage"
            className="hidden"
            accept="image/*"
            onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])}
          />
        </label>
      )}
    </div>
  );
}

function ResultCard({ result }: { result: any }) {
  if (result.found) {
    const info = result.dogInfo;
    const confidence = result.confidence ?? 0;
    const genderMap: Record<string, string> = { male: "公", female: "母" };

    return (
      <div className="animate-fade-in-up">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-emerald-100 rounded-full flex items-center justify-center">
            <svg className="w-5 h-5 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
          </div>
          <div>
            <h3 className="font-bold text-emerald-700 text-lg">成功识别！</h3>
            <p className="text-sm text-gray-500">已在数据库中找到匹配的犬只档案</p>
          </div>
        </div>

        {/* Confidence */}
        <div className="bg-gray-50 rounded-xl p-4 mb-6">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-medium text-gray-600">匹配置信度</span>
            <span className="text-sm font-bold text-blue-600">{confidence.toFixed(1)}%</span>
          </div>
          <div className="w-full h-2.5 bg-gray-200 rounded-full overflow-hidden">
            <div
              className="h-full rounded-full bg-gradient-to-r from-blue-400 to-blue-600"
              style={{ width: `${Math.min(confidence, 100)}%` }}
            />
          </div>
        </div>

        {/* Dog Info */}
        <h4 className="font-semibold text-gray-800 mb-4">犬只档案信息</h4>
        <div className="grid grid-cols-2 gap-4">
          {[
            { label: "名字", value: info?.name ?? "未知", icon: "🏷️" },
            { label: "性别", value: genderMap[info?.gender] ?? "未知", icon: "⚥" },
            { label: "年龄", value: info?.age ?? "未知", icon: "📅" },
            { label: "品种", value: info?.breed ?? "未知", icon: "🐕" },
          ].map((item, i) => (
            <div key={i} className="bg-white border border-gray-100 rounded-xl p-4 hover:border-blue-100 transition-colors">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-base">{item.icon}</span>
                <span className="text-xs text-gray-400 font-medium uppercase tracking-wide">{item.label}</span>
              </div>
              <p className="font-semibold text-gray-900 text-base">{item.value}</p>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="animate-fade-in-up">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 bg-red-100 rounded-full flex items-center justify-center">
          <svg className="w-5 h-5 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </div>
        <div>
          <h3 className="font-bold text-red-600 text-lg">未找到匹配记录</h3>
          <p className="text-sm text-gray-500">数据库中暂无与该鼻纹匹配的犬只档案</p>
        </div>
      </div>

      <div className="bg-amber-50 border border-amber-100 rounded-xl p-4 mb-4">
        <p className="text-amber-800 text-sm font-medium mb-2">建议您：</p>
        <ul className="text-amber-700 text-sm space-y-1.5">
          <li className="flex items-center gap-2">
            <svg className="w-4 h-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
            使用正面清晰、光线充足的照片
          </li>
          <li className="flex items-center gap-2">
            <svg className="w-4 h-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
            确认该犬只是否已在系统中登记
          </li>
          <li className="flex items-center gap-2">
            <svg className="w-4 h-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
            <Link href="/register" className="underline underline-offset-2 font-medium">前往登记新犬只</Link>
          </li>
        </ul>
      </div>
    </div>
  );
}

export default function SearchDog() {
  const [noseImage, setNoseImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any | null>(null);

  const handleFile = (file: File) => {
    setNoseImage(file);
    const reader = new FileReader();
    reader.onloadend = () => setPreview(reader.result as string);
    reader.readAsDataURL(file);
  };

  const handleReset = () => {
    setNoseImage(null);
    setPreview("");
    setResult(null);
  };

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!noseImage) return;
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("noseImage", noseImage);
      const response = await axios.post("/api/dogs/search", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(response.data);
    } catch (error) {
      console.error("查询失败:", error);
      alert("查询失败，请检查网络连接或稍后重试");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      <NavBar />

      <main className="max-w-4xl mx-auto px-4 sm:px-6 py-10">
        {/* Page Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-extrabold text-gray-900 mb-2">查询犬类信息</h1>
          <p className="text-gray-500">上传犬鼻纹照片，自动在数据库中检索匹配记录</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
          {/* Left: Upload + Search */}
          <div className="lg:col-span-3 space-y-5">
            <div className="bg-white rounded-2xl p-6 border border-gray-100 shadow-sm">
              <h2 className="font-bold text-gray-800 mb-4 flex items-center gap-2">
                <svg className="w-5 h-5 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14" />
                </svg>
                上传鼻纹图片
              </h2>
              <UploadZone preview={preview} onFile={handleFile} onReset={handleReset} />
            </div>

            <form onSubmit={handleSearch} className="flex gap-3">
              <button
                type="submit"
                className="btn-primary flex-1 gap-2"
                disabled={loading || !noseImage}
              >
                {loading ? (
                  <>
                    <span className="spinner" />
                    <span>识别中...</span>
                  </>
                ) : (
                  <>
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                        d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                    开始识别
                  </>
                )}
              </button>
              {(noseImage || result) && (
                <button type="button" onClick={handleReset} className="btn-secondary gap-2">
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                      d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                  重置
                </button>
              )}
            </form>

            {/* Tips */}
            {!result && (
              <div className="bg-blue-50 border border-blue-100 rounded-xl p-4">
                <p className="text-blue-800 text-sm font-medium mb-2">拍摄建议</p>
                <ul className="text-blue-700 text-sm space-y-1">
                  <li>• 在自然光或充足人工光源下拍摄</li>
                  <li>• 保持相机与鼻面平行，避免斜角</li>
                  <li>• 确保鼻纹纹路清晰可见，无遮挡</li>
                </ul>
              </div>
            )}
          </div>

          {/* Right: Result */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-2xl p-6 border border-gray-100 shadow-sm h-full min-h-[300px]">
              {result ? (
                <ResultCard result={result} />
              ) : (
                <div className="flex flex-col items-center justify-center h-full text-center text-gray-400 py-12">
                  <div className="w-16 h-16 bg-gray-100 rounded-2xl flex items-center justify-center mb-4">
                    <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1}
                        d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                  </div>
                  <p className="font-medium text-gray-500 mb-1">识别结果</p>
                  <p className="text-sm">上传图片并点击「开始识别」后，结果将显示在此处</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
