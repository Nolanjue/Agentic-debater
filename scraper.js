const puppeteer = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');
const axios = require('axios');
const pdf = require('pdf-parse');

puppeteer.use(StealthPlugin());

async function scrapePDFs(query) {
    const browser = await puppeteer.launch({ headless: true });
    const page = await browser.newPage();
    
    try {
        await page.goto(`https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=${query}&btnG=`, { waitUntil: 'networkidle2' });


        // Extract links to PDFs from the search results
        // Wait for search results to load
    await page.waitForSelector('.gs_r .gs_ggsd a');

    // Extract PDF links
    const pdfLinks = await page.evaluate(() => {
        const results = document.querySelectorAll('.gs_r .gs_ggsd a');
        const links = [];
        
        results.forEach(result => {
            const href = result.getAttribute('href');
            if (href && href.endsWith('.pdf')) {
                links.push(href);
            }
        });

        return links;
    });

    let compiled_text = [];
    for (const [index, pdfLink] of pdfLinks.entries()) {
        try {
            const text = await fetchAndParsePDF(pdfLink);
            console.log(`Text content of source ${pdfLink}:`);
            
            //REMEMBER THAT SUBPROCESS TAKES PRINT STATEMENTS AS ITS OUTPUT AS IT BASICALLY RUNS THE PROGRAM AND RETURNS THE OUTPUT OF ANYTHING
            console.log(text);
            compiled_text.push(text);

        } catch (error) {
            console.error(`Error parsing PDF from ${pdfLink}:`, error);
            
            continue;
        }
    }
    return compiled_text

    } catch (error) {
        console.error('Error during scraping:', error);
    } finally {
        await browser.close();
    }
}

// Usage: node scholar_pdf_scraper.js "your search query"
const args = process.argv.slice(2);
const query = args.join(' ');
scrapePDFs(query);


async function fetchAndParsePDF(url) {
    const response = await axios({
        url,
        method: 'GET',
        responseType: 'arraybuffer'
    });
    const dataBuffer = response.data;
    const data = await pdf(dataBuffer);
    return data.text.substring(0, 10000); // Limiting to the first 10000 characters
}