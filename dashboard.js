var thrift = require('thrift');
var cvService = require('./gen-nodejs/cvService.js');
var ttypes = require('./gen-nodejs/cv_types.js');
var thriftConnection = thrift.createConnection('127.0.0.1', 9999);
var thriftClient = thrift.createClient(cvService, thriftConnection);

thriftConnection.on("error",function(e)
{
    console.log(e);
});

var targetLocation = "";
var map_targetLocation = "";
var liveFrameUpdatePeriod = 100; //100ms

var map_image_json = {
    "template": "<figure class='text-center'><div class='img-container thumbnail'><img src='<%this.img_src%>'></div><figcaption class='text-bold'><%this.caption%></figcaption></figure>",
    "data": []
}
function verification(signal, light, countdown){
    if(signal == "Red"){
        $(light).removeClass("bg-yellow");
        $(light).addClass("bg-red");
        document.querySelector(light+"text").innerHTML = "NA";
    }
    else if(signal == "Green"){
        $(light).removeClass("bg-red");
        $(light).addClass("bg-green");
        document.querySelector(light+"text").innerHTML = countdown;
    }
    else if(signal == "Yellow"){
        $(light).removeClass("bg-green");
        $(light).addClass("bg-yellow");
        document.querySelector(light+"text").innerHTML = countdown;
    }
}

function changeMAP(signal){
    var green = signal.indexOf("Green")
    var yellow = signal.indexOf("Yellow")
    if (green== -1){
        document.getElementById("light_map").src = "light/"+String(yellow)+"yellow.jpg";
    }
    else if (yellow == -1){
        document.getElementById("light_map").src = "light/"+String(green)+"green.jpg";
    }
}
setInterval(function(){
    map_targetLocation = get_location();
    if (map_targetLocation != ""){
        thriftClient.get_location_info(map_targetLocation, (error, result) => {
            if(error) {
                console.error(error);
            }
            else
            {
                images = result.encoded_images
                map_image_json['data'] = [];
                camera_id = 1
                for (var key in images) {
                    image  = images[key];
                    imgSrc = "data:image/jpg;base64,"+image;
                    map_image_json['data'].push(
                        {
                            "img_src": imgSrc,
                            "caption": key
                        }
                    )
                    // document.querySelector('#camera'+camera_id).setAttribute("src", imgSrc);
                    camera_id += 1;
                }
                changeMAP(result.traffic_signals)
                var signalColor_1 = result.traffic_signals[0];
                var signalColor_2 = result.traffic_signals[1];
                var signalColor_3 = result.traffic_signals[2];
                var signalColor_4 = result.traffic_signals[3];
                var signalColor_5 = result.traffic_signals[4];
                document.querySelector('#map-pedestrians').innerHTML = result.n_pedestrians;
                document.querySelector('#map-vehicles').innerHTML = result.n_vehicles;
                var countdown = result.count_down
                // document.querySelector('#map-traffic-light-count-down').innerHTML = result.count_down;
                verification(signalColor_1, '#light1',countdown)
                verification(signalColor_2, '#light2',countdown)
                verification(signalColor_3, '#light3',countdown)
                verification(signalColor_4, '#light4',countdown)
                verification(signalColor_5, '#p-light',countdown)
            }
        });

        if (map_image_json['data'].length > 0){
            var list = $('#map-list-videos').data('list');
            list._createItemsFromJSON(map_image_json);
            list.draw();
        }

    }
}, liveFrameUpdatePeriod);