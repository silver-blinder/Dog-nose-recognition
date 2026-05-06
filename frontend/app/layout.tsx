import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: '犬鼻纹识别系统 | Dog Nose Print Recognition',
  description: '基于孪生神经网络的犬只身份识别平台，通过犬鼻纹实现精准比对与信息管理',
  keywords: '犬鼻纹,犬只识别,孪生网络,深度学习,宠物身份',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="zh-CN">
      <body className="antialiased min-h-screen bg-gray-50">
        {children}
      </body>
    </html>
  )
}
