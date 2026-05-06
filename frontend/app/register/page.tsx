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
      className={`relative w-full h-64 rounded-2xl border-2 border-dashed transition-all duration-300 overflow-hidden
        ${dragging ? "border-emerald-500 bg-emerald-50 scale-[1.01]" : "border-gray-200 bg-gray-50 hover:border-emerald-300 hover:bg-emerald-50/40"}
        ${preview ? "border-solid border-emerald-200 bg-white" : "cursor-pointer"}`}
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
            <span className="badge-success text-xs">图片已选择</span>
          </div>
        </>
      ) : (
        <label htmlFor="noseImageReg" className="flex flex-col items-center justify-center h-full cursor-pointer p-8 text-center">
          <div className="w-16 h-16 bg-emerald-100 rounded-2xl flex items-center justify-center mb-4">
            <svg className="w-8 h-8 text-emerald-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
          </div>
          <p className="text-gray-700 font-semibold text-lg mb-1">点击上传或拖拽图片</p>
          <p className="text-gray-400 text-sm">支持 JPG、PNG 格式</p>
          <input
            type="file"
            id="noseImageReg"
            className="hidden"
            accept="image/*"
            onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])}
          />
        </label>
      )}
    </div>
  );
}

// ── Duplicate Found View ─────────────────────────────────────────────────────
function DuplicateView({
  result,
  dogInfo,
  loading,
  onChange,
  onUpdate,
  onReset,
}: {
  result: any;
  dogInfo: any;
  loading: boolean;
  onChange: (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => void;
  onUpdate: () => void;
  onReset: () => void;
}) {
  const existing = result.bestMatch?.dogInfo;
  const genderMap: Record<string, string> = { male: "公", female: "母" };

  return (
    <div className="space-y-6 animate-fade-in-up">
      {/* Warning Banner */}
      <div className="flex items-start gap-4 bg-amber-50 border border-amber-200 rounded-2xl p-5">
        <div className="w-10 h-10 bg-amber-100 rounded-full flex items-center justify-center flex-shrink-0">
          <svg className="w-5 h-5 text-amber-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        </div>
        <div>
          <h3 className="font-bold text-amber-800 mb-1">该犬只已存在于数据库中</h3>
          <p className="text-amber-700 text-sm">系统检测到与该鼻纹高度匹配的已有记录，如需更新信息请在下方编辑后提交。</p>
        </div>
      </div>

      {/* Existing Info */}
      <div className="bg-white rounded-2xl border border-gray-100 p-6">
        <h4 className="font-bold text-gray-800 mb-4">已有档案信息</h4>
        <div className="grid grid-cols-2 gap-3">
          {[
            { label: "名字", value: existing?.name, icon: "🏷️" },
            { label: "性别", value: genderMap[existing?.gender] ?? existing?.gender, icon: "⚥" },
            { label: "年龄", value: existing?.age, icon: "📅" },
            { label: "品种", value: existing?.breed, icon: "🐕" },
          ].map((item, i) => (
            <div key={i} className="bg-gray-50 rounded-xl p-3">
              <p className="text-xs text-gray-400 mb-0.5">{item.icon} {item.label}</p>
              <p className="font-semibold text-gray-800">{item.value ?? "—"}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Update Form */}
      <div className="bg-white rounded-2xl border border-gray-100 p-6">
        <h4 className="font-bold text-gray-800 mb-4">更新档案信息（选填）</h4>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-600 mb-1.5">名字</label>
            <input name="name" value={dogInfo.name} onChange={onChange}
              className="input-field" placeholder={existing?.name ?? "输入新名字"} />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-600 mb-1.5">性别</label>
            <div className="relative">
              <select name="gender" value={dogInfo.gender} onChange={onChange} className="select-field pr-10">
                <option value="">请选择</option>
                <option value="male">公</option>
                <option value="female">母</option>
              </select>
              <svg className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none"
                   fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </div>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-600 mb-1.5">年龄</label>
            <input name="age" value={dogInfo.age} onChange={onChange}
              className="input-field" placeholder={existing?.age ?? "如：2岁"} />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-600 mb-1.5">品种</label>
            <input name="breed" value={dogInfo.breed} onChange={onChange}
              className="input-field" placeholder={existing?.breed ?? "如：金毛寻回犬"} />
          </div>
        </div>
        <div className="flex gap-3 mt-5">
          <button onClick={onUpdate} className="btn-success gap-2 flex-1" disabled={loading}>
            {loading ? <><span className="spinner" /><span>更新中...</span></> : "更新档案信息"}
          </button>
          <button onClick={onReset} className="btn-secondary gap-2">重新登记</button>
        </div>
      </div>
    </div>
  );
}

// ── Success View ─────────────────────────────────────────────────────────────
function SuccessView({ result, onReset }: { result: any; onReset: () => void }) {
  const info = result.dogInfo;
  const genderMap: Record<string, string> = { male: "公", female: "母" };

  return (
    <div className="space-y-6 animate-fade-in-up">
      <div className="flex items-start gap-4 bg-emerald-50 border border-emerald-200 rounded-2xl p-5">
        <div className="w-10 h-10 bg-emerald-100 rounded-full flex items-center justify-center flex-shrink-0">
          <svg className="w-5 h-5 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
        </div>
        <div>
          <h3 className="font-bold text-emerald-800 mb-1">登记成功！</h3>
          <p className="text-emerald-700 text-sm">该犬只已成功加入数据库，档案信息如下。</p>
        </div>
      </div>

      <div className="bg-white rounded-2xl border border-gray-100 p-6">
        <h4 className="font-bold text-gray-800 mb-4">新增档案信息</h4>
        <div className="grid grid-cols-2 gap-3">
          {[
            { label: "名字", value: info?.name, icon: "🏷️" },
            { label: "性别", value: genderMap[info?.gender] ?? info?.gender, icon: "⚥" },
            { label: "年龄", value: info?.age, icon: "📅" },
            { label: "品种", value: info?.breed, icon: "🐕" },
          ].map((item, i) => (
            <div key={i} className="bg-emerald-50 rounded-xl p-3">
              <p className="text-xs text-emerald-600 mb-0.5">{item.icon} {item.label}</p>
              <p className="font-semibold text-gray-800">{item.value ?? "—"}</p>
            </div>
          ))}
        </div>
      </div>

      <button onClick={onReset} className="btn-primary w-full gap-2">
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
        </svg>
        继续登记新犬只
      </button>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
export default function RegisterDog() {
  const [noseImage, setNoseImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any | null>(null);
  const [showDogInfo, setShowDogInfo] = useState(false);
  const [dogInfo, setDogInfo] = useState({ name: "", gender: "", age: "", breed: "" });

  const handleFile = (file: File) => {
    setNoseImage(file);
    const reader = new FileReader();
    reader.onloadend = () => setPreview(reader.result as string);
    reader.readAsDataURL(file);
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setDogInfo((prev) => ({ ...prev, [name]: value }));
  };

  const handleReset = () => {
    setNoseImage(null);
    setPreview("");
    setResult(null);
    setDogInfo({ name: "", gender: "", age: "", breed: "" });
    setShowDogInfo(false);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!noseImage) { alert("请先上传鼻纹图片"); return; }
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("noseImage", noseImage);
      formData.append("name", dogInfo.name || "未命名");
      formData.append("gender", dogInfo.gender || "unknown");
      formData.append("age", dogInfo.age || "未知");
      formData.append("breed", dogInfo.breed || "未知");
      const response = await axios.post("/api/dogs/create", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(response.data);
    } catch {
      alert("登记失败，请检查网络连接或稍后重试");
    } finally {
      setLoading(false);
    }
  };

  const handleUpdate = async () => {
    if (!result?.bestMatch?.dogInfo?.id) { alert("找不到要更新的犬只ID"); return; }
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("dogId", result.bestMatch.dogInfo.id);
      formData.append("name", dogInfo.name || result.bestMatch.dogInfo.name);
      formData.append("gender", dogInfo.gender || result.bestMatch.dogInfo.gender);
      formData.append("age", dogInfo.age || result.bestMatch.dogInfo.age);
      formData.append("breed", dogInfo.breed || result.bestMatch.dogInfo.breed);
      await axios.put("/api/dogs/update", formData);
      alert("档案信息更新成功！");
    } catch {
      alert("更新失败，请稍后重试");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-emerald-50/30 to-indigo-50">
      <NavBar />

      <main className="max-w-4xl mx-auto px-4 sm:px-6 py-10">
        {/* Page Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-extrabold text-gray-900 mb-2">登记新犬只</h1>
          <p className="text-gray-500">为您的爱犬建立鼻纹档案，实现永久身份标识</p>
        </div>

        {result ? (
          result.is_duplicate ? (
            <DuplicateView
              result={result}
              dogInfo={dogInfo}
              loading={loading}
              onChange={handleChange}
              onUpdate={handleUpdate}
              onReset={handleReset}
            />
          ) : (
            <SuccessView result={result} onReset={handleReset} />
          )
        ) : (
          <form onSubmit={handleSubmit} className="space-y-5">
            <div className="grid grid-cols-1 lg:grid-cols-5 gap-5">
              {/* Upload */}
              <div className="lg:col-span-3">
                <div className="bg-white rounded-2xl p-6 border border-gray-100 shadow-sm">
                  <h2 className="font-bold text-gray-800 mb-4 flex items-center gap-2">
                    <svg className="w-5 h-5 text-emerald-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                        d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                    </svg>
                    上传鼻纹照片
                    <span className="text-red-400 text-lg">*</span>
                  </h2>
                  <UploadZone preview={preview} onFile={handleFile} onReset={() => { setNoseImage(null); setPreview(""); }} />
                </div>
              </div>

              {/* Dog Info */}
              <div className="lg:col-span-2">
                <div className="bg-white rounded-2xl border border-gray-100 shadow-sm overflow-hidden">
                  <button
                    type="button"
                    onClick={() => setShowDogInfo(!showDogInfo)}
                    className="w-full flex items-center justify-between px-6 py-4 hover:bg-gray-50 transition-colors"
                  >
                    <div className="flex items-center gap-2">
                      <svg className="w-5 h-5 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                          d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                      </svg>
                      <span className="font-bold text-gray-800">犬只基本信息</span>
                      <span className="text-xs text-gray-400 font-normal">选填</span>
                    </div>
                    <svg className={`w-5 h-5 text-gray-400 transition-transform duration-200 ${showDogInfo ? "rotate-180" : ""}`}
                         fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </button>

                  {showDogInfo && (
                    <div className="px-6 pb-6 space-y-4 border-t border-gray-100 pt-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-600 mb-1.5">名字</label>
                        <input name="name" value={dogInfo.name} onChange={handleChange}
                          className="input-field" placeholder="如：大黄" />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-600 mb-1.5">性别</label>
                        <div className="relative">
                          <select name="gender" value={dogInfo.gender} onChange={handleChange} className="select-field pr-10">
                            <option value="">请选择</option>
                            <option value="male">公</option>
                            <option value="female">母</option>
                          </select>
                          <svg className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none"
                               fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                          </svg>
                        </div>
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-600 mb-1.5">年龄</label>
                        <input name="age" value={dogInfo.age} onChange={handleChange}
                          className="input-field" placeholder="如：3岁" />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-600 mb-1.5">品种</label>
                        <input name="breed" value={dogInfo.breed} onChange={handleChange}
                          className="input-field" placeholder="如：拉布拉多" />
                      </div>
                    </div>
                  )}

                  {!showDogInfo && (
                    <div className="px-6 pb-5">
                      <p className="text-sm text-gray-400">点击展开填写犬只详细信息（可跳过）</p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Submit */}
            <div className="flex gap-3">
              <button
                type="submit"
                className="btn-success flex-1 gap-2 text-base py-3.5"
                disabled={loading || !noseImage}
              >
                {loading ? (
                  <><span className="spinner" /><span>提交中...</span></>
                ) : (
                  <>
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                    </svg>
                    提交登记
                  </>
                )}
              </button>
              <button type="button" onClick={handleReset} className="btn-secondary gap-2">
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                重置
              </button>
            </div>

            {/* Notice */}
            <div className="bg-blue-50 border border-blue-100 rounded-xl p-4 text-sm text-blue-700">
              <strong>提示：</strong>提交后，系统将自动检测数据库中是否已有相同犬只的鼻纹记录。
              若检测到重复，您可以选择更新现有档案信息。
            </div>
          </form>
        )}
      </main>
    </div>
  );
}
