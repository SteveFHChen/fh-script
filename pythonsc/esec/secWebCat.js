天天基金网数据接口
一、接口

所有基金公司名称列表代码
http://fund.eastmoney.com/js/jjjz_gs.js

所有基金名称列表代码
http://fund.eastmoney.com/js/fundcode_search.js

股票代码的编码规则
http://www.360doc.com/content/15/0308/11/76571_453496598.shtml
http://www.xunyang.com.cn/cgjq/5812.html

基金详细信息和历史数据：
http://fund.eastmoney.com/pingzhongdata/161725.js

获取基金实时数据：
http://fundgz.1234567.com.cn/js/161725.js
返回值：jsonpgz({"fundcode":"161725","name":"招商中证白酒指数(LOF)","jzrq":"2021-01-28","dwjz":"1.3997","gsz":"1.4258","gszzl":"1.86","gztime":"2021-01-29 15:00"});

获取股票实时数据：
http://api.money.126.net/data/feed/0600519,0601318,1000333,0600036,money.api
http://hq.sinajs.cn/list=sh600519,sh601318,s_sz000333,s_sh600036

用Python爬取历年基金数据
https://blog.csdn.net/fei347795790/article/details/102638769

美股代码查询一览表：
http://quote.eastmoney.com/usstocklist.html

中国股票网-股票代码大全
http://www.yz21.org/stock/info/

股票代码一览表，最全的中国股市代码大全
股票代码分类：
http://www.xunyang.com.cn/cgjq/5812.html

天天基金网-行情中心
沪深个股/板块/指数、港股、美股、英股、全球指数、期货、基金、债券、外汇、期权、黄金
http://quote.eastmoney.com/center/hszs.html

凤凰网-财经-股票/基金/债券列表
A股、B股、港股、基金、债券
http://app.finance.ifeng.com/hq/list.php?type=hkstock

搜狐证券
指数
深沪AB股排行（涨跌幅榜、高低价股、震幅榜、成交量、成交额）
板块（板块综合、行业分类、地域板块、概念板块）
https://q.stock.sohu.com/cn/ph.shtml

基金持仓比例
http://www.51sanhu.com/aspx/funds.aspx?fcode=&gcode=&index=14

获取股票历史数据（网易163行情接口）
https://blog.csdn.net/weixin_44766484/article/details/105802794
	网易原始财经接口清单1：

	日内实时盘口（JSON）：
	[http://api.money.126.net/data/feed/1000002,1000001,1000881,0601398,money.api]

	历史成交数据（CSV）：
	[http://quotes.money.163.com/service/chddata.html?code=0601398&start=20000720&end=20150508]

	财务指标（CSV）：
	[http://quotes.money.163.com/service/zycwzb_601398.html?type=report]

	资产负债表（CSV）：
	[http://quotes.money.163.com/service/zcfzb_601398.html]

	利润表（CSV）：
	[http://quotes.money.163.com/service/lrb_601398.html]

	现金流表（CSV）：
	[http://quotes.money.163.com/service/xjllb_601398.html]

	杜邦分析（HTML）：
	[http://quotes.money.163.com/f10/dbfx_601398.html]

数据接口-免费版（股票数据API）
https://blog.csdn.net/u011331731/article/details/101353475

实时股票数据接口大全
https://www.cnblogs.com/rockchip/p/3182525.html
https://www.cnblogs.com/jackljf/p/3589216.html


import requests
import json
import re

code = "161725"  # 基金代码
url = "http://fundgz.1234567.com.cn/js/%s.js"%code
# 浏览器头
headers = {'content-type': 'application/json',
           'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:22.0) Gecko/20100101 Firefox/22.0'}

r = requests.get(url, headers=headers)
# 返回信息
content = r.text
# content = """jsonpgz({"fundcode":"161725","name":"招商中证白酒指数(LOF)","jzrq":"2021-01-28","dwjz":"1.3997","gsz":"1.4258","gszzl":"1.86","gztime":"2021-01-29 15:00"});"""

#解析方法1：parse to json/dictionary
fundRealInfo = json.loads(content[len('jsonpgz('):-len(');')])

#解析方法2: split string by pattern
# 正则表达式
pattern = r'^jsonpgz\((.*)\)'
# 查找结果
search = re.findall(pattern, content)
# 遍历结果
for i in search:
  data = json.loads(i)
  # print(data,type(data))
  print("基金: {},收益率: {}".format(data['name'],data['gsz']))


fs = []
A2Z = document.querySelectorAll("ul.search-left li a")
for(let i=1; i<A2Z.length; i++){
	let A2Zi = A2Z[i]
	A2Zi.click()
	
	fc0 = document.querySelectorAll("tr[data-al] td.seaCol6")
	fc0.forEach((oc, ic, ac)=>{
		//如何实现鼠标悬停改变焦点
		oc.click()
		
		fs0 = document.querySelectorAll("div.CP_content ul li a")
		fs0.forEach((of, fi, af)=>{
			fs.push({
				"A2Z": A2Zi.innerText, 
				"companName": oc.innerText, 
				"fundCode":of.getAttribute("data-cookie"), 
				"fundName": of.innerText})
		})
	})
}


A2Z = document.querySelectorAll("ul.search-left li a")
for(let i=1; i<A2Z.length; i++){
	let A2Zi = A2Z[i]
	A2Zi.click()
}

fc = []
fc0 = document.querySelectorAll("tr[data-al='Z'] td.seaCol6")
fc0.forEach((o, i, a)=>{
	fc.push({"name": o.innerText})
})

fs = []
fs0 = document.querySelectorAll("div.CP_content ul li a")
fs0.forEach((o, i, a)=>{
	fs.push({"code":o.getAttribute("data-cookie"), "name": o.innerText})
})


pages = document.querySelectorAll("div.pagestock a")
return pages[pages.length-1]

stockList = []
stockRows = document.querySelectorAll("table.stockBlock tr")
stockRows.forEach(function(s){
	if(s.childNodes[2].innerText!="股票代码")
		stockList.push({
			"code": s.childNodes[2].innerText,
			"name": s.childNodes[3].innerText,
			"company": s.childNodes[4].innerText
		});
});

