import { createClient } from "@supabase/supabase-js";
import { NextResponse } from "next/server";

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
);

export async function PUT(request: Request) {
  try {
    const formData = await request.formData();
    const dogId = formData.get("dogId") as string;

    if (!dogId) {
      return NextResponse.json({ error: "缺少狗狗ID" }, { status: 400 });
    }

    const name = formData.get("name") as string;
    const gender = formData.get("gender") as string;
    const age = formData.get("age") as string;
    const breed = formData.get("breed") as string;

    const { data, error } = await supabase
      .from("dogs")
      .update({
        name,
        gender,
        age,
        breed,
        updated_at: new Date().toISOString(),
      })
      .eq("id", dogId)
      .select();

    if (error) {
      console.error("更新狗狗信息失败:", error);
      return NextResponse.json({ error: "更新狗狗信息失败" }, { status: 500 });
    }

    return NextResponse.json({
      success: true,
      message: "狗狗信息已更新",
      dogInfo: data?.[0] || null,
    });
  } catch (error) {
    console.error("处理失败:", error);
    return NextResponse.json({ error: "处理请求失败" }, { status: 500 });
  }
}
