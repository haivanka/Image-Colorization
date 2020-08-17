$(document).ready( function() {
    context = document.getElementById('canvas').getContext("2d");

    var color = "#ff0000"

    var clickX = new Array();
    var clickY = new Array();

    context.lineWidth = 5;
    context.fillStyle = color;
    context.fill();


    var paint;

    $('#canvas').mousedown(function(e){
        var mouseX = e.pageX - $(this).offset().left
        var mouseY = e.pageY - $(this).offset().top;

        context.fillRect(mouseX, mouseY, 10 , 10);
    });

    document.getElementById('color').addEventListener('change', changeColor);

    function changeColor() {
        context.fillStyle = this.value;
    }

    $(document).on('change', '.btn-file :file', function() {
		var input = $(this),
			label = input.val().replace(/\\/g, '/').replace(/.*\//, '');
		     input.trigger('fileselect', [label]);
		});

		$('.btn-file :file').on('fileselect', function(event, label) {

		    var input = $(this).parents('.input-group').find(':text'),
		        log = label;

		    if( input.length ) {
		        input.val(log);
		    } else {
		        if( log ) alert(log);
		    }

		});
		function readURL(input) {
		    if (input.files && input.files[0]) {
		        var reader = new FileReader();

		        reader.onload = function (e) {

		            $('#img-upload').attr('src', e.target.result);
		        }

		        reader.readAsDataURL(input.files[0]);
		    }
		}

		$("#image").change(function(){
		    readURL(this);
		});
	});