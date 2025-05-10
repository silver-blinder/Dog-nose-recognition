"use client";

import axios from "axios";
import Link from "next/link";
import { useState } from "react";

export default function SearchDog() {
  const [noseImage, setNoseImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any | null>(null);

  const handleImageUpload = (file: File) => {
    setNoseImage(file);
    const reader = new FileReader();
    reader.onloadend = () => {
      setPreview(reader.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handleResetImage = () => {
    setNoseImage(null);
    setPreview("");
  };

  const handleReset = () => {
    setNoseImage(null);
    setPreview("");
    setResult(null);
  };

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!noseImage) {
      alert("请上传狗鼻子的照片");
      return;
    }

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("noseImage", noseImage);

      const response = await axios.post("/api/dogs/search", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      setResult(response.data);
    } catch (error) {
      console.error("查询失败:", error);
      alert("查询失败，请重试");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen p-8 flex flex-col items-center justify-center bg-gray-50">
      <main className="w-full max-w-3xl">
        <div className="text-center mb-12">
          <h1 className="text-3xl font-bold mb-2 text-black">查询犬类信息</h1>
          <p className="text-gray-600">上传狗鼻子的照片查询犬类信息</p>
          <div className="mt-4">
            <Link href="/" className="text-blue-500 hover:underline">
              返回首页
            </Link>
          </div>
        </div>

        {!result ? (
          <form onSubmit={handleSearch} className="space-y-6">
            <div className="bg-white p-6 rounded-lg shadow-sm">
              <h2 className="text-xl font-semibold mb-4 text-black">上传照片</h2>
              <div className="flex justify-center">
                <div className="flex flex-col items-center relative">
                  <label
                    className="w-[300px] h-[300px] border-2 border-dashed border-gray-300 rounded-lg flex items-center justify-center bg-white hover:border-blue-500 transition-colors cursor-pointer relative group"
                    htmlFor="noseImage"
                  >
                    {preview ? (
                      <div className="w-full h-full flex items-center justify-center relative">
                        <img
                          src={preview}
                          alt="预览图"
                          className="w-[250px] h-[250px] object-contain"
                        />
                        <button
                          onClick={(e) => {
                            e.preventDefault();
                            handleResetImage();
                          }}
                          className="absolute top-2 right-2 w-8 h-8 bg-black bg-opacity-50 rounded-full text-white flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                        >
                          ×
                        </button>
                      </div>
                    ) : (
                      <div className="text-center p-4">
                        <p className="text-gray-500">点击上传狗鼻子图片</p>
                        <p className="text-gray-400 text-sm mt-2">支持 JPG、PNG 格式</p>
                      </div>
                    )}
                  </label>
                  <input
                    type="file"
                    id="noseImage"
                    className="hidden"
                    accept="image/*"
                    onChange={(e) => e.target.files?.[0] && handleImageUpload(e.target.files[0])}
                  />
                </div>
              </div>
            </div>

            <div className="flex justify-center gap-4">
              <button
                type="submit"
                className="bg-blue-500 text-white px-8 py-3 rounded-lg hover:bg-blue-600 transition-colors disabled:bg-gray-300 disabled:cursor-not-allowed"
                disabled={loading || !noseImage}
              >
                {loading ? "查询中..." : "查询"}
              </button>
              <button
                type="button"
                className="bg-gray-500 text-white px-8 py-3 rounded-lg hover:bg-gray-600 transition-colors"
                onClick={handleReset}
              >
                重置
              </button>
            </div>
          </form>
        ) : (
          <div className="bg-white p-6 rounded-lg shadow-sm">
            <h2 className="text-xl font-semibold mb-4 text-black">查询结果</h2>

            {result.found ? (
              <div className="space-y-4">
                <p className="text-lg text-green-600">成功找到匹配的狗狗！</p>
                <div className="p-4 bg-gray-50 rounded-md">
                  <h3 className="text-lg font-medium mb-2">狗狗信息：</h3>
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <p className="text-sm text-gray-500">名字</p>
                      <p>{result.dogInfo.name}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">性别</p>
                      <p>
                        {result.dogInfo.gender === "male"
                          ? "公"
                          : result.dogInfo.gender === "female"
                          ? "母"
                          : "未知"}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">年龄</p>
                      <p>{result.dogInfo.age}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">品种</p>
                      <p>{result.dogInfo.breed}</p>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <p className="text-lg text-red-600">未找到匹配的狗狗</p>
                <p className="text-gray-600">
                  无法在数据库中找到与上传照片匹配的狗鼻纹。您可以尝试：
                </p>
                <ul className="list-disc list-inside text-gray-600 ml-4">
                  <li>上传更清晰的照片</li>
                  <li>确认该犬只是否已在系统中登记</li>
                  <li>
                    <Link href="/register" className="text-blue-500 hover:underline">
                      登记新犬鼻纹
                    </Link>
                  </li>
                </ul>
              </div>
            )}

            <div className="mt-6 flex justify-center">
              <button
                onClick={handleReset}
                className="bg-blue-500 text-white px-8 py-3 rounded-lg hover:bg-blue-600 transition-colors"
              >
                继续查询
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
