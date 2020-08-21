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

                document.querySelector('#map-pedestrians').innerHTML = result.n_pedestrians;
                document.querySelector('#map-vehicles').innerHTML = result.n_vehicles;
                document.querySelector('#map-traffic-light-count-down').innerHTML = result.count_down;

            }
        });

        if (map_image_json['data'].length > 0){
            var list = $('#map-list-videos').data('list');
            list._createItemsFromJSON(map_image_json);
            list.draw();
        }

        // $('#light1').removeClass("bg-cyan");
        // $('#light2').removeClass("bg-green");
        // $('#light3').removeClass("bg-orange");
        // $('#light4').removeClass("bg-red");
        //
        // var signalColor = result.traffic_signals.toUpperCase();
        // console.log(signalColor)
        // if(signalColor == "GREEN"){
        //     $('#traffic-signal-bg').addClass("bg-green");
        // }
        // else if(signalColor == "YELLOW"){
        //     $('#traffic-signal-bg').addClass("bg-amber");
        // }
        // else if(signalColor == "RED"){
        //     $('#traffic-signal-bg').addClass("bg-red");
        // }
        // else{
        //     $('#traffic-signal-bg').addClass("bg-dark");
        // }
    }
}, liveFrameUpdatePeriod);