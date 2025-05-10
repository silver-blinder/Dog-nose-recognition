"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";

export default function Home() {
  const router = useRouter();

  return (
    <div className="min-h-screen p-8 flex flex-col items-center justify-center bg-gray-50">
      <main className="w-full max-w-3xl">
        <div className="text-center mb-12">
          <h1 className="text-3xl font-bold mb-2 text-black">犬鼻纹识别系统</h1>
          <p className="text-gray-600">通过犬鼻纹识别和管理您的犬类信息</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div
            className="bg-white p-8 rounded-lg shadow-sm border border-gray-200 flex flex-col items-center hover:border-blue-500 hover:shadow-md transition-all cursor-pointer"
            onClick={() => router.push("/search")}
          >
            <div className="w-24 h-24 mb-4 bg-blue-100 rounded-full flex items-center justify-center">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-12 w-12 text-blue-500"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                />
              </svg>
            </div>
            <h2 className="text-xl font-semibold mb-2 text-black">查询犬类信息</h2>
            <p className="text-gray-600 text-center">上传犬鼻纹，快速查询犬类的详细信息</p>
            <Link
              href="/search"
              className="mt-6 bg-blue-500 text-white px-6 py-2 rounded-lg hover:bg-blue-600 transition-colors inline-block"
            >
              立即查询
            </Link>
          </div>

          <div
            className="bg-white p-8 rounded-lg shadow-sm border border-gray-200 flex flex-col items-center hover:border-green-500 hover:shadow-md transition-all cursor-pointer"
            onClick={() => router.push("/register")}
          >
            <div className="w-24 h-24 mb-4 bg-green-100 rounded-full flex items-center justify-center">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-12 w-12 text-green-500"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 6v6m0 0v6m0-6h6m-6 0H6"
                />
              </svg>
            </div>
            <h2 className="text-xl font-semibold mb-2 text-black">登记新犬类</h2>
            <p className="text-gray-600 text-center">添加新的犬鼻纹和犬类信息到数据库</p>
            <Link
              href="/register"
              className="mt-6 bg-green-500 text-white px-6 py-2 rounded-lg hover:bg-green-600 transition-colors inline-block"
            >
              开始登记
            </Link>
          </div>
        </div>

        <div className="mt-12 p-6 bg-white rounded-lg shadow-sm">
          <h2 className="text-xl font-semibold mb-4 text-black">关于犬鼻纹识别</h2>
          <p className="text-gray-600">
            犬鼻纹识别是一种有效的犬类身份识别方式，类似于人类指纹识别。每只狗的鼻纹图案都是独一无二的，
            通过我们的系统，您可以快速查询或登记犬类信息，帮助更好地管理和识别犬只身份。
          </p>
        </div>
      </main>
    </div>
  );
}
