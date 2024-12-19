"use client";

import { useState } from "react";
import axios from "axios";

export default function Home() {
  const [image1, setImage1] = useState<File | null>(null);
  const [image2, setImage2] = useState<File | null>(null);
  const [preview1, setPreview1] = useState<string>("");
  const [preview2, setPreview2] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{
    is_same_dog: boolean;
    confidence: number;
  } | null>(null);


  const handleImageUpload = (
    file: File,
    setImage: (file: File) => void,
    setPreview: (url: string) => void
  ) => {
    setImage(file);
    const reader = new FileReader();
    reader.onloadend = () => {
      setPreview(reader.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handleCompare = async () => {
    if (!image1 || !image2) {
      alert("请上传两张图片");
      return;
    }

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("images", image1);
      formData.append("images", image2);

      const response = await axios.post("/api/compare", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      setResult(response.data);
    } catch (error) {
      console.error("比对失败:", error);
      alert("比对失败，请重试");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen p-8 flex flex-col items-center justify-center bg-gray-50">
      <main className="w-full max-w-4xl">
        {/* 标题部分 */}
        <div className="text-center mb-12">
          <h1 className="text-3xl font-bold mb-2 text-black">犬鼻纹识别系统</h1>
          <p className="text-gray-600">上传两张狗鼻子的照片，系统将判断是否为同一只狗</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="flex flex-col items-center">
            <label
              className="w-full h-64 border-2 border-dashed border-gray-300 rounded-lg flex items-center justify-center bg-white hover:border-blue-500 transition-colors cursor-pointer"
              htmlFor="image1"
            >
              {preview1 ? (
                <img
                  src={preview1}
                  alt="预览图1"
                  className="max-h-full max-w-full object-contain"
                />
              ) : (
                <div className="text-center p-4">
                  <p className="text-gray-500">点击上传第一张图片</p>
                  <p className="text-gray-400 text-sm mt-2">支持 JPG、PNG 格式</p>
                </div>
              )}
            </label>
            <input
              type="file"
              id="image1"
              className="hidden"
              accept="image/*"
              onChange={(e) =>
                e.target.files?.[0] && handleImageUpload(e.target.files[0], setImage1, setPreview1)
              }
            />
          </div>

          <div className="flex flex-col items-center">
            <label
              className="w-full h-64 border-2 border-dashed border-gray-300 rounded-lg flex items-center justify-center bg-white hover:border-blue-500 transition-colors cursor-pointer"
              htmlFor="image2"
            >
              {preview2 ? (
                <img
                  src={preview2}
                  alt="预览图2"
                  className="max-h-full max-w-full object-contain"
                />
              ) : (
                <div className="text-center p-4">
                  <p className="text-gray-500">点击上传第二张图片</p>
                  <p className="text-gray-400 text-sm mt-2">支持 JPG、PNG 格式</p>
                </div>
              )}
            </label>
            <input
              type="file"
              id="image2"
              className="hidden"
              accept="image/*"
              onChange={(e) =>
                e.target.files?.[0] && handleImageUpload(e.target.files[0], setImage2, setPreview2)
              }
            />
          </div>
        </div>

        <div className="mt-8 flex justify-center">
          <button
            className={`bg-blue-500 text-white px-8 py-3 rounded-lg hover:bg-blue-600 transition-colors disabled:bg-gray-300 disabled:cursor-not-allowed`}
            onClick={handleCompare}
            disabled={!image1 || !image2 || loading}
          >
            {loading ? "比对中..." : "开始比对"}
          </button>
        </div>

        <div className="mt-12 p-6 bg-white rounded-lg shadow-sm">
          <h2 className="text-xl font-semibold mb-4">比对结果</h2>
          {result ? (
            <div>
              <p className="text-lg">
                {result.is_same_dog ? (
                  <span className="text-green-600">✓ 这是同一只狗</span>
                ) : (
                  <span className="text-red-600">✗ 这不是同一只狗</span>
                )}
              </p>
              <p className="text-gray-600 mt-2">置信度：{result.confidence}%</p>
            </div>
          ) : (
            <p className="text-gray-600">上传两张照片后即可查看比对结果</p>
          )}
        </div>
      </main>
    </div>
  );
}
