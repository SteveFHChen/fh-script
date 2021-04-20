let mainBox = document.getElementById("mainBox");

sqlList.forEach(function(header, index){
	//console.log(header.headline)
	let divSQL = document.createElement("div");
	divSQL.className = "cSQL";
	divSQL.style.border = "2px solid #088";
	
	let divHeader = document.createElement("div")
	divHeader.className = "cHeader";
	divHeader.innerText = index + " - " + header.headline.replace("#@#", "");
	divHeader.onclick = function(a){
	//divHeader.ondblclick = function(a){
		console.log(a.target.innerHTML);
		let divSQL = a.target.parentElement;
		//t1 = a;
		//console.log(divSQL.querySelector("textarea").style.display);
		if(divSQL.querySelector("textarea").style.display=="block"){
			divSQL.querySelector("textarea").style.display="none";
		}else{
			divSQL.querySelector("textarea").style.display="block";
		}
	}
	divSQL.append(divHeader);
	
	let sqlArea = document.createElement("textarea");
	sqlArea.innerHTML = header.content.join("\n");
	//console.log(sqlArea.innerText);
	
	sqlArea.addEventListener("input", function(e) {
	  this.style.height = "inherit";
	  this.style.height = `${this.scrollHeight}px`;
	});
	divSQL.append(sqlArea);
	
	mainBox.append(divSQL);
})
