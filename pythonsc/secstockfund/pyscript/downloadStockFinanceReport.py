#Download stock finance report info
from fhsechead import *
import execjs

currentDate = time.strftime("%Y%m%d", time.localtime())
savePath="C:/fh/testenv1/sec/"

logFile='C:/fh/testenv1/sec/stockFinanceReport.log'
fLog = LogWriter(logFile)

driver = webdriver.Chrome(executable_path=chromeDriverExe, chrome_options=chrome_options)
driver.set_page_load_timeout(40)


#Capture stock finance indexes, and fill into DB
def captureStockFinanceIndexes(stockCode):
    try:
        #driver.get("http://f10.eastmoney.com/NewFinanceAnalysis/Index?type=web&code=SH600519#") #Sample
        driver.get(f"http://f10.eastmoney.com/NewFinanceAnalysis/Index?type=web&code={stockCode}#") 
    except:
        fLog.writeLog(f"Loading page timeout, stock is {stockCode}.")
    
    jsStockFinanceIndexes = """
        function formatValue(datax, c){
            data = datax.trim();
            result = data;
            try{
                if(data=="" || data=="--"){
                    result = null;
                }else if(!isNaN(data)){
                    result = Number(result)
                }else if(data.endsWith("亿")){
                    result = data.substring(0, data.length-"亿".length)
                    result = Number(result) * 100000000;
                }else if (data.endsWith("万")){
                    result = data.substring(0, data.length-"万".length)
                    result = Number(result) * 10000
                }
            }catch(err){
                console.error(err);
                result = data;
            }
            return result;
        }

        //Unit test
        /*
        console.log(formatValue("467.0亿"))
        console.log(formatValue("467.0万"))
        console.log(formatValue("-467.0万"))
        console.log(formatValue("20-12-31"))
        console.log(formatValue("--"))
        console.log(formatValue("467.0"))
        console.log(formatValue("  "))
        console.log(formatValue(""))
        */

        financeMainIndexes = []
        financeMainIndexesElements = document.querySelectorAll("div.content#report_zyzb table tbody tr")
        for(let r=0; r<financeMainIndexesElements.length; r++){
            rowIndexes = financeMainIndexesElements[r].children
            financeMainIndex = []
            for(let c=0; c<rowIndexes.length; c++){
                cellValue = rowIndexes[c].innerText.trim();
                if(c!=0){
                    cellValue = formatValue(cellValue, c)
                }
                financeMainIndex.push(cellValue)
                //console.log(rowIndexes[c].innerText.trim());
            }
            financeMainIndexes.push(financeMainIndex)
        }

        return JSON.stringify(financeMainIndexes);
    """
    financeIndexes=json.loads(driver.execute_script(jsStockFinanceIndexes))
    print(financeIndexes)
    
if __name__=="__main__":
    
    captureStockFinanceIndexes("SH600519")
    
    cs1.close()
    conn.close()
    driver.close()
    fLog.close()

