var mymap = L.map('mapid').setView([22.3527359, 114.1460038], 11.75);
// var location = "";
//https://leaflet-extras.github.io/leaflet-providers/preview/
var Jawg_Matrix = L.tileLayer('https://{s}.tile.jawg.io/jawg-light/{z}/{x}/{y}{r}.png?access-token={accessToken}', {
	attribution: '<a href="http://jawg.io" title="Tiles Courtesy of Jawg Maps" target="_blank">&copy; <b>Jawg</b>Maps</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
	minZoom: 0,
	maxZoom: 22,
	subdomains: 'abcd',
	accessToken: '9WzbZYz2kn0LEpUv7DRNvJVnD6wleEei1v0Y3tga4RomIBvBLzHDi98b2Iy2zM41'
}).addTo(mymap);


//https://github.com/pointhi/leaflet-color-markers
var greenIcon = new L.Icon({
	iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-green.png',
	shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
	iconSize: [25, 41],
	iconAnchor: [12, 41],
	popupAnchor: [1, -34],
	shadowSize: [41, 41]
  });

var redIcon = new L.Icon({
	iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
	shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
	iconSize: [25, 41],
	iconAnchor: [12, 41],
	popupAnchor: [1, -34],
	shadowSize: [41, 41]
  });

function onMarkerClick(e) {
	alert(this.getLatLng());
	console.log(this);
}

var locations = ""
popupST = createPopup("LaiKing")
STpop = L.marker([22.346054, 114.129087], {icon: redIcon}).addTo(mymap)
	.bindPopup(popupST['customHTML'], popupST['customOptions']);
// .on('click', onMarkerClick); //mouseover
STpop.on('click', function(e) {
	locations = "Location-A";
})

popupTY = createPopup("IVE (Tsing Yi)")
TYpop = L.marker([22.342375, 114.106237], {icon: redIcon}).addTo(mymap)
	.bindPopup(popupTY['customHTML'], popupTY['customOptions']);
TYpop.on('click', function(e) {
	locations = "Location-B";
})
	// .on('click', onMarkerClick); //mouseover

popupHW = createPopup("IVE (Haking Wong)")
HWpop = L.marker([22.335466, 114.152343], {icon: redIcon}).addTo(mymap)
	.bindPopup(popupHW['customHTML'], popupHW['customOptions']);
HWpop.on('click', function(e) {
	locations = "Location-C";
})
	// .on('click', onMarkerClick); //mouseover

function get_location(){
	return locations;
}

function createPopup(location){
	var customHTML = "NA";
	var customOptions =
		{
			'minWidth': '250',
			'maxWidth': '500',
			'className' : 'custom'
		};

	if (location=="IVE (Sha Tin)"){
		customHTML = "IVE (Sha Tin)<br/><br/><img src='https://engineering.vtc.edu.hk/images/IVE-ST.jpg'/>"}
	else if(location=="IVE (Tsing Yi)"){
		customHTML = "IVE (Tsing Yi)<br/><br/><img src='https://engineering.vtc.edu.hk/images/IVE-TY.jpg'/>";
	}
	else if(location=="IVE (Haking Wong)"){
		customHTML = "IVE (Haking Wong)<br/><br/><img src='https://engineering.vtc.edu.hk/images/IVE-HW.jpg'/>";
	}

	return {"customHTML":customHTML, "customOptions":customOptions}
}
