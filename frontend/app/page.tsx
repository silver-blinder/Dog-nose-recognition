'use client'

import axios from 'axios'
import { useState } from 'react'

export default function Home() {
  const [image1, setImage1] = useState<File | null>(null)
  const [image2, setImage2] = useState<File | null>(null)
  const [preview1, setPreview1] = useState<string>('')
  const [preview2, setPreview2] = useState<string>('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<{
    is_same_dog: boolean
    confidence: number
  } | null>(null)

  const handleImageUpload = (
    file: File,
    setImage: (file: File) => void,
    setPreview: (url: string) => void,
  ) => {
    setImage(file)
    const reader = new FileReader()
    reader.onloadend = () => {
      setPreview(reader.result as string)
    }
    reader.readAsDataURL(file)
  }

  const handleResetImage = (
    setImage: (file: File | null) => void,
    setPreview: (url: string) => void,
  ) => {
    setImage(null)
    setPreview('')
  }

  const handleReset = () => {
    setImage1(null)
    setImage2(null)
    setPreview1('')
    setPreview2('')
    setResult(null)
  }

  const handleCompare = async () => {
    if (!image1 || !image2) {
      alert('请上传两张图片')
      return
    }

    setLoading(true)
    try {
      const formData = new FormData()
      formData.append('images', image1)
      formData.append('images', image2)

      const response = await axios.post('/api/compare', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })

      setResult(response.data)
    }
    catch (error) {
      console.error('比对失败:', error)
      alert('比对失败，请重试')
    }
    finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen p-8 flex flex-col items-center justify-center bg-gray-50">
      <main className="w-full max-w-3xl">
        <div className="text-center mb-12">
          <h1 className="text-3xl font-bold mb-2 text-black">犬鼻纹识别系统</h1>
          <p className="text-gray-600">上传两张狗鼻子的照片，系统将判断是否为同一只狗</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="flex flex-col items-center relative">
            <label
              className="w-[300px] h-[300px] border-2 border-dashed border-gray-300 rounded-lg flex items-center justify-center bg-white hover:border-blue-500 transition-colors cursor-pointer relative group"
              htmlFor="image1"
            >
              {preview1
                ? (
                  <div className="w-full h-full flex items-center justify-center relative">
                    <img
                      src={preview1}
                      alt="预览图1"
                      className="w-[250px] h-[250px] object-contain"
                    />
                    <button
                      onClick={(e) => {
                        e.preventDefault()
                        handleResetImage(setImage1, setPreview1)
                      }}
                      className="absolute top-2 right-2 w-8 h-8 bg-black bg-opacity-50 rounded-full text-white flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                    >
                      ×
                    </button>
                  </div>
                )
                : (
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
              onChange={e =>
                e.target.files?.[0] && handleImageUpload(e.target.files[0], setImage1, setPreview1)}
            />
          </div>

          <div className="flex flex-col items-center relative">
            <label
              className="w-[300px] h-[300px] border-2 border-dashed border-gray-300 rounded-lg flex items-center justify-center bg-white hover:border-blue-500 transition-colors cursor-pointer relative group"
              htmlFor="image2"
            >
              {preview2
                ? (
                  <div className="w-full h-full flex items-center justify-center relative">
                    <img
                      src={preview2}
                      alt="预览图2"
                      className="w-[250px] h-[250px] object-contain"
                    />
                    <button
                      onClick={(e) => {
                        e.preventDefault()
                        handleResetImage(setImage2, setPreview2)
                      }}
                      className="absolute top-2 right-2 w-8 h-8 bg-black bg-opacity-50 rounded-full text-white flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                    >
                      ×
                    </button>
                  </div>
                )
                : (
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
              onChange={e =>
                e.target.files?.[0] && handleImageUpload(e.target.files[0], setImage2, setPreview2)}
            />
          </div>
        </div>

        <div className="mt-8 flex justify-center gap-4">
          <button
            className="bg-blue-500 text-white px-8 py-3 rounded-lg hover:bg-blue-600 transition-colors disabled:bg-gray-300 disabled:cursor-not-allowed"
            onClick={handleCompare}
            disabled={!image1 || !image2 || loading}
          >
            {loading ? '比对中...' : '开始比对'}
          </button>

          <button
            className="w-[128px] bg-gray-500 text-white px-8 py-3 rounded-lg hover:bg-gray-600 transition-colors disabled:bg-gray-300 disabled:cursor-not-allowed"
            onClick={handleReset}
            disabled={loading || (!image1 && !image2 && !result)}
          >
            重置
          </button>
        </div>

        <div className="mt-12 p-6 bg-white rounded-lg shadow-sm">
          <h2 className="text-xl font-semibold mb-4 text-black">比对结果</h2>
          {result
            ? (
              <div>
                <p className="text-lg">
                  {result.is_same_dog
                    ? (
                      <span className="text-green-600">✓ 这是同一只狗</span>
                    )
                    : (
                      <span className="text-red-600">✗ 这不是同一只狗</span>
                    )}
                </p>
                <p className="text-gray-600 mt-2">
                  置信度：
                  {result.confidence}
                  %
                </p>
              </div>
            )
            : (
              <p className="text-gray-600">上传两张照片后即可查看比对结果</p>
            )}
        </div>
      </main>
    </div>
  )
}
