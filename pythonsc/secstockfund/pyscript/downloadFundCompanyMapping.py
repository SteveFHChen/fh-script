#Capture fund and company mapping, and fill into DB
from fhsechead import *

logFile='C:/fh/testenv1/sec/fund-updatefundtrend.log'
fLog = LogWriter(logFile)

driver = webdriver.Chrome(executable_path="C:/fh/testenv1/chromedriver.exe")
driver.set_page_load_timeout(40)

def captureFundCompanyMapping():
    sqlCompnayList = "SELECT company_code, company_name FROM sec_fund_company"
    rowcount = cs1.execute(sqlCompnayList)
    fundCompanyList = cs1.fetchall()
    for company in fundCompanyList:
        #company = fundCompanyList[0]
        companyCode, companyName = company[0], company[1]
        try:
            driver.get(f"http://fund.eastmoney.com/Company/{companyCode}.html") 
        except:
            fLog.writeLog(f'Loading page timeout, company is {company}.')
        
        jsFundCompanyMapping = f"""
            fundCompanyMapping = {{"companyCode":"{companyCode}", "fundList":[]}}
            document.querySelectorAll("a.code").forEach(function(f, i, a){{
                fundCompanyMapping.fundList.push(f.innerText);
            }});
            return JSON.stringify(fundCompanyMapping);
            """
        fundCompanyMapping=json.loads(driver.execute_script(jsFundCompanyMapping))
        fundListString = "'"+"','".join(fundCompanyMapping['fundList'])+"'"
        sqlUpdateFundCompanyMapping = f"""
            UPDATE sec_fund
            SET company_code = '{companyCode}'
            WHERE fund_code in ({fundListString})
            """
        rowcount = cs1.execute(sqlUpdateFundCompanyMapping)
        conn.commit()
        fLog.writeLog(f"Company {companyCode}-{companyName} has {len(fundCompanyMapping['fundList'])} funds in internet, updated {rowcount} funds in DB.")

    fLog.writeLog(f"Update company:fund mapping for {len(fundCompanyList)} companies completed.")

if __name__ == "__main__":
    captureFundCompanyMapping()
    
    cs1.close()
    conn.close()
    driver.close()
    fLog.close()