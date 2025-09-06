import json
from playwright.async_api import async_playwright

async def extract_links(url):
    links = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, timeout=60000)
        await page.wait_for_load_state("networkidle")

        anchors = await page.query_selector_all("a")
        for a in anchors:
            text = (await a.inner_text()).strip()
            href = await a.get_attribute("href")
            if href:
                if href.startswith("#"):
                    continue
                elif href.startswith("/"):
                    href = page.url.rstrip("/") + href
                links.append({"text": text, "href": href})

        await browser.close()
    return links

async def main():
    url = "https://mc.mos.ru/info/trp-rp"
    data = await extract_links(url)
    with open("data/raw/links.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"已保存 {len(data)} 个链接到 data/raw/links.json")

if __name__ == "__main__":
    main()
