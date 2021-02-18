from fhsechead import *

logFile='C:/fh/testenv1/sec/stock.log'
fLog = LogWriter(logFile)

serieses = {
    "sh":{"code":"sh", "sectionType":"上证", "description":"上证系列指数"}, 
    "sz":{"code":"sz", "sectionType":"深证", "description":"深证系列指数"}, 
    "components":{"code":"components", "sectionType":"成份", "description":"指数成份"}, 
    "zzzs":{"code":"zzzs", "sectionType":"中证", "description":"中证系列指数"}
    }

def captureCNIndexes(series):
    fLog.writeLog(f"Start to capture CN indexes {series['description']} ...")
    driver = webdriver.Chrome(executable_path=chromeDriverExe, chrome_options=chrome_options)
    driver.set_page_load_timeout(40)

    try:
        driver.get(f"http://quote.eastmoney.com/center/gridlist.html#index_{series['code']}") # get接受url可以是如何网址，此处以百度为例
    except:
        fLog.writeLog(f'Loading page timeout.')

    jsPageNumber = """
        maxPage = document.querySelector("a.paginate_button:nth-last-child(1)")
        return parseInt(maxPage.innerText)
        """
    pageNumber = driver.execute_script(jsPageNumber)
    
    jsCatCNIndexList = """
        indexList = []
        indexRows = document.querySelectorAll(".listview.full #table_wrapper-table tbody tr")
        indexRows.forEach(function(r, i, a){
            indexList.push({
                /* stockType: '指数', */
                code: r.children[1].innerText,
                name: r.children[2].innerText
            });
        });
        return JSON.stringify(indexList);
        """
    #pageNumber = 2
    for i in range(1, pageNumber+1):
    #for i in range(2, 4+1): #For testing
        fLog.writeLog(f"Start capture page [{i}]...")

        pageInput = driver.find_element(By.CSS_SELECTOR,"input.paginate_input")
        goButton = driver.find_element(By.CSS_SELECTOR,"a.paginte_go")

        pageInput.clear()
        pageInput.send_keys(i)
        goButton.click()
        
        time.sleep(3)
        
        cnIndexList=json.loads(driver.execute_script(jsCatCNIndexList))
        for s in cnIndexList:
            sqlKeyCheck = f"""
                SELECT 
                    stock_code 
                FROM sec_stock 
                WHERE stock_code = '{s['code']}' AND stock_type = '指数'
                """
            sqlAdd = f"""
                INSERT INTO sec_stock(stock_code, stock_name, region_code, section_name, stock_type)
                VALUES ('{s['code']}', '{s['name']}', 'CN', '{series['sectionType']}', '指数')
                """
            try:
                rowcount = cs1.execute(sqlKeyCheck)
                if rowcount>0:
                    fLog.writeLog(f"Stock already exists {s['code']} - {s['name']}")
                else:
                    rowcount = cs1.execute(sqlAdd)
            except Exception as e:
                fLog.writeLog(f"Failed add: {s}")
                fLog.writeLog(f"Exception msg: {e}")
        conn.commit()
        fLog.writeLog(f"Completed capture page [{i}].")
    fLog.writeLog(f"Finished capture CN stock indexes {series['description']}.")
    driver.close()

if __name__ == "__main__":
    captureCNIndexes(serieses['sh'])
    captureCNIndexes(serieses['sz'])
    #captureCNIndexes(serieses['components']) #Same as sh + sz
    captureCNIndexes(serieses['zzzs'])
    cs1.close()
    conn.close()
    fLog.close()