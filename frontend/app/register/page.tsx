"use client";

import axios from "axios";
import Link from "next/link";
import { useState } from "react";

export default function RegisterDog() {
  const [noseImage, setNoseImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any | null>(null);
  const [showDogInfo, setShowDogInfo] = useState(false);
  const [dogInfo, setDogInfo] = useState({
    name: "",
    gender: "",
    age: "",
    breed: "",
  });

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

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setDogInfo((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!noseImage) {
      alert("请上传狗鼻子的照片");
      return;
    }

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("noseImage", noseImage);

      // Add dog info with defaults if not provided
      formData.append("name", dogInfo.name || "未命名");
      formData.append("gender", dogInfo.gender || "unknown");
      formData.append("age", dogInfo.age || "未知");
      formData.append("breed", dogInfo.breed || "未知");

      const response = await axios.post("/api/dogs/create", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      setResult(response.data);
    } catch (error) {
      console.error("注册失败:", error);
      alert("注册失败，请重试");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setNoseImage(null);
    setPreview("");
    setResult(null);
    setDogInfo({
      name: "",
      gender: "",
      age: "",
      breed: "",
    });
  };

  const handleUpdateDogInfo = async () => {
    if (!result?.bestMatch?.dogInfo?.id) {
      alert("找不到要更新的狗狗ID");
      console.error("Missing dog ID:", result?.bestMatch);
      return;
    }

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("dogId", result.bestMatch.dogInfo.id);
      formData.append("name", dogInfo.name || result.bestMatch.dogInfo.name);
      formData.append("gender", dogInfo.gender || result.bestMatch.dogInfo.gender);
      formData.append("age", dogInfo.age || result.bestMatch.dogInfo.age);
      formData.append("breed", dogInfo.breed || result.bestMatch.dogInfo.breed);

      await axios.put("/api/dogs/update", formData);

      alert("狗狗信息更新成功");
      handleReset();
    } catch (error) {
      console.error("更新失败:", error);
      alert("更新失败，请重试");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen p-8 flex flex-col items-center justify-center bg-gray-50">
      <main className="w-full max-w-3xl">
        <div className="text-center mb-12">
          <h1 className="text-3xl font-bold mb-2 text-black">登记新犬鼻纹</h1>
          <p className="text-gray-600">上传狗鼻子的照片和基本信息进行登记</p>
          <div className="mt-4">
            <Link href="/" className="text-blue-500 hover:underline">
              返回首页
            </Link>
          </div>
        </div>

        {!result ? (
          <form onSubmit={handleSubmit} className="space-y-6">
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

            <div className="bg-white p-6 rounded-lg shadow-sm">
              <button
                type="button"
                onClick={() => setShowDogInfo(!showDogInfo)}
                className="flex items-center justify-between w-full text-left text-xl font-semibold mb-2 text-black"
              >
                <span>狗狗信息 (选填)</span>
                <span className="text-lg">{showDogInfo ? "▲" : "▼"}</span>
              </button>

              {showDogInfo && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">名字</label>
                    <input
                      type="text"
                      name="name"
                      value={dogInfo.name}
                      onChange={handleInputChange}
                      className="w-full p-2 border border-gray-300 rounded-md"
                      placeholder="未命名"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">性别</label>
                    <select
                      name="gender"
                      value={dogInfo.gender}
                      onChange={handleInputChange}
                      className="w-full p-2 border border-gray-300 rounded-md"
                    >
                      <option value="">请选择</option>
                      <option value="male">公</option>
                      <option value="female">母</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">年龄</label>
                    <input
                      type="text"
                      name="age"
                      value={dogInfo.age}
                      onChange={handleInputChange}
                      className="w-full p-2 border border-gray-300 rounded-md"
                      placeholder="未知"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">品种</label>
                    <input
                      type="text"
                      name="breed"
                      value={dogInfo.breed}
                      onChange={handleInputChange}
                      className="w-full p-2 border border-gray-300 rounded-md"
                      placeholder="未知"
                    />
                  </div>
                </div>
              )}
            </div>

            <div className="flex justify-center gap-4">
              <button
                type="submit"
                className="bg-blue-500 text-white px-8 py-3 rounded-lg hover:bg-blue-600 transition-colors disabled:bg-gray-300 disabled:cursor-not-allowed"
                disabled={loading}
              >
                {loading ? "提交中..." : "提交"}
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
            <h2 className="text-xl font-semibold mb-4 text-black">检查结果</h2>

            {result.is_duplicate ? (
              <div className="space-y-4">
                <p className="text-lg text-red-600">该狗已存在于数据库中！</p>
                <div className="p-4 bg-gray-50 rounded-md">
                  <h3 className="text-lg font-medium mb-2">已存在的狗狗信息：</h3>
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <p className="text-sm text-gray-500">名字</p>
                      <p>{result.bestMatch.dogInfo.name}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">性别</p>
                      <p>
                        {result.bestMatch.dogInfo.gender === "male"
                          ? "公"
                          : result.bestMatch.dogInfo.gender === "female"
                          ? "母"
                          : "未知"}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">年龄</p>
                      <p>{result.bestMatch.dogInfo.age}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">品种</p>
                      <p>{result.bestMatch.dogInfo.breed}</p>
                    </div>
                  </div>
                </div>

                <div className="p-4 bg-gray-50 rounded-lg">
                  <h3 className="text-lg font-medium mb-4">更新狗狗信息</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">名字</label>
                      <input
                        type="text"
                        name="name"
                        value={dogInfo.name}
                        onChange={handleInputChange}
                        className="w-full p-2 border border-gray-300 rounded-md"
                        placeholder={result.bestMatch.dogInfo.name}
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">性别</label>
                      <select
                        name="gender"
                        value={dogInfo.gender}
                        onChange={handleInputChange}
                        className="w-full p-2 border border-gray-300 rounded-md"
                      >
                        <option value="">请选择</option>
                        <option value="male">公</option>
                        <option value="female">母</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">年龄</label>
                      <input
                        type="text"
                        name="age"
                        value={dogInfo.age}
                        onChange={handleInputChange}
                        className="w-full p-2 border border-gray-300 rounded-md"
                        placeholder={result.bestMatch.dogInfo.age}
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">品种</label>
                      <input
                        type="text"
                        name="breed"
                        value={dogInfo.breed}
                        onChange={handleInputChange}
                        className="w-full p-2 border border-gray-300 rounded-md"
                        placeholder={result.bestMatch.dogInfo.breed}
                      />
                    </div>
                  </div>
                  <div className="mt-4 flex justify-center">
                    <button
                      onClick={handleUpdateDogInfo}
                      className="bg-green-500 text-white px-6 py-2 rounded-lg hover:bg-green-600 transition-colors disabled:bg-gray-300 disabled:cursor-not-allowed"
                      disabled={loading}
                    >
                      {loading ? "更新中..." : "更新信息"}
                    </button>
                  </div>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <p className="text-lg text-green-600">成功添加新狗狗到数据库！</p>
                <div className="p-4 bg-gray-50 rounded-md">
                  <h3 className="text-lg font-medium mb-2">新增狗狗信息：</h3>
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
            )}

            <div className="mt-6 flex justify-center">
              <button
                onClick={handleReset}
                className="bg-blue-500 text-white px-8 py-3 rounded-lg hover:bg-blue-600 transition-colors"
              >
                继续登记
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
