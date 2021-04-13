;(function () {
    
    'use strict';

    // iPad and iPod detection  
    var isiPad = function(){
        return (navigator.platform.indexOf("iPad") != -1);
    };

    var isiPhone = function(){
        return (
            (navigator.platform.indexOf("iPhone") != -1) || 
            (navigator.platform.indexOf("iPod") != -1)
        );
    };

    // OffCanvass
    var offCanvass = function() {
        $('body').on('click', '.js-fh5co-menu-btn, .js-fh5co-offcanvass-close', function(){
            $('#fh5co-offcanvass').toggleClass('fh5co-awake');
        });
    };

    // Click outside of offcanvass
    var mobileMenuOutsideClick = function() {
        $(document).click(function (e) {
        var container = $("#fh5co-offcanvass, .js-fh5co-menu-btn");
        if (!container.is(e.target) && container.has(e.target).length === 0) {
            if ( $('#fh5co-offcanvass').hasClass('fh5co-awake') ) {
                $('#fh5co-offcanvass').removeClass('fh5co-awake');
            }
        }
        });

        $(window).scroll(function(){
            if ( $(window).scrollTop() > 500 ) {
                if ( $('#fh5co-offcanvass').hasClass('fh5co-awake') ) {
                    $('#fh5co-offcanvass').removeClass('fh5co-awake');
                }
            }
        });
    };

    // Magnific Popup
    
    var magnifPopup = function() {
        $('.image-popup').magnificPopup({
            type: 'image',
            removalDelay: 300,
            mainClass: 'mfp-with-zoom',
            titleSrc: 'title',
            gallery:{
                enabled:true
            },
            zoom: {
                enabled: true, // By default it's false, so don't forget to enable it

                duration: 300, // duration of the effect, in milliseconds
                easing: 'ease-in-out', // CSS transition easing function

                // The "opener" function should return the element from which popup will be zoomed in
                // and to which popup will be scaled down
                // By defailt it looks for an image tag:
                opener: function(openerElement) {
                // openerElement is the element on which popup was initialized, in this case its <a> tag
                // you don't need to add "opener" option if this code matches your needs, it's defailt one.
                return openerElement.is('img') ? openerElement : openerElement.find('img');
                }
            },
            disableOn: function() {
                if( selecting || swapping ) {
                    return false;
                }
                return true;
            }
        });
    };



    var animateBoxWayPoint = function() {

        if ($('.animate-box').length > 0) {
            $('.animate-box').waypoint( function( direction ) {

                if( direction === 'down' && !$(this).hasClass('animated') ) {
                    $(this.element).addClass('bounceIn animated');
                }

            } , { offset: '75%' } );
        }

    };

    

    
    $(function(){
        magnifPopup();
        offCanvass();
        mobileMenuOutsideClick();
        animateBoxWayPoint();
    });

    /* Change layout to show images properly
    Yichun Shi
    */
    var selecting = false;
    var swapping = 0; // 0: off 1: waiting for first 2: waiting for second
    var swapping_element = null;

    var turnOnSelect = function () {
        selecting = true;
        $('#navbar-menu').addClass('no-shadow');
        $('#navbar-menu li').addClass('disabled');
        $('#navbar-menu li a').addClass('disabled');
        $('#navbar-selection').addClass('active');

        $('a.fh5co-board-img').on('click', function () {
            return false;
        });
        $('a.fh5co-board-img img').on('mouseup', function () {
            $(this).toggleClass("selected");
        });
    }

    var turnOffSelect = function() {
        selecting = false;
        $('#navbar-menu').removeClass('no-shadow');
        $('#navbar-menu li').removeClass('disabled');
        $('#navbar-menu li a').removeClass('disabled');
        $('#navbar-selection').removeClass('active');
        $('a.fh5co-board-img img').off('mouseup');
        $('a.fh5co-board-img img.selected').removeClass('selected');
    }

    var turnOnSwap = function () {
        swapping = 1;
        $('#navbar-menu').addClass('no-shadow');
        $('#navbar-menu li').addClass('disabled');
        $('#navbar-menu li a').addClass('disabled');
        $('#navbar-swap').addClass('active');

        $('a.fh5co-board-img').on('click', function () {
            return false;
        });
        $('a.fh5co-board-img img').on('mouseup', function () {
            if (swapping == 1) {
                $(this).addClass("swapping");
                swapping_element = $(this).parent().parent();
                swapping ++;
            } else if (swapping == 2) {
                swapElements($(swapping_element), $(this).parent().parent());
                clearSwap();
            }
        });
    }

    var clearSwap = function () {
        swapping = 1;
        swapping_element = null;
        $('a.fh5co-board-img img.swapping').removeClass('swapping');
    }

    var turnOffSwap = function() {
        swapping = 0;
        swapping_element = null;
        $('#navbar-menu').removeClass('no-shadow');
        $('#navbar-menu li').removeClass('disabled');
        $('#navbar-menu li a').removeClass('disabled');
        $('#navbar-swap').removeClass('active');
        $('a.fh5co-board-img img').off('mouseup');
        $('a.fh5co-board-img img.swapping').removeClass('swapping');
    }

    var deleteEmptyGroups = function() {
        $('#fh5co-board .item').each(function() {
            if($(this).has('a.fh5co-board-img').length==0) {
                $(this).remove();
            }
        });
    }

    var swapElements = function($s, $t) {
        var $m1 = $("<a></a>");
        var $m2 = $("<a></a>");
        $s.before($m1);
        $t.before($m2);
        $m1.before($t);
        $m2.before($s);
        $m1.remove();
        $m2.remove();
        // $s.before($t.clone());
        // $t.before($s);
        // $t.remove();
    }

    $(function(){
        salvattore.rescanMediaQueries();

        /***  Main Menu ***/
        $('#btn-1col').on('click', function () {
            $('[data-columns]').attr('data-before','1 .column.size-1of1');
            salvattore.rescanMediaQueries();
        });
        $('#btn-2col').on('click', function () {
            $('[data-columns]').attr('data-before','2 .column.size-1of2');
            salvattore.rescanMediaQueries();
        });
        $('#btn-3col').on('click', function () {
            $('[data-columns]').attr('data-before','3 .column.size-1of3');
            salvattore.rescanMediaQueries();
        });
        $('#btn-4col').on('click', function () {
            $('[data-columns]').attr('data-before','4 .column.size-1of4');
            salvattore.rescanMediaQueries();
        });
        $('#btn-6col').on('click', function () {
            $('[data-columns]').attr('data-before','6 .column.size-1of6');
            salvattore.rescanMediaQueries();
        });
        $('#btn-8col').on('click', function () {
            $('[data-columns]').attr('data-before','8 .column.size-1of8');
            salvattore.rescanMediaQueries();
        });
        $('#btn-12col').on('click', function () {
            $('[data-columns]').attr('data-before','12 .column.size-1of12');
            salvattore.rescanMediaQueries();
        });
        $('#btn-1col-img').on('click', function () {
            $('a.fh5co-board-img img').attr('class','size-1of1');
            salvattore.rescanMediaQueries();
        });
        $('#btn-2col-img').on('click', function () {
            $('a.fh5co-board-img img').attr('class','size-1of2');
            salvattore.rescanMediaQueries();
        });
        $('#btn-3col-img').on('click', function () {
            $('a.fh5co-board-img img').attr('class','size-1of3');
            salvattore.rescanMediaQueries();
        });
        $('#btn-4col-img').on('click', function () {
            $('a.fh5co-board-img img').attr('class','size-1of4');
            salvattore.rescanMediaQueries();
        });
        $('#btn-8col-img').on('click', function () {
            $('a.fh5co-board-img img').attr('class','size-1of8');
            salvattore.rescanMediaQueries();
        });
        $('#btn-view-desc').on('click', function () {
            $(this).toggleClass('active');
            $('div.fh5co-desc').toggleClass('hidden');
            $('#btn-view-desc').toggleClass('active');
        });
        $('#btn-view-bcg').on('click', function () {
            $(this).toggleClass('active');
            $('body').toggleClass('no-bcg');
            $('#btn-view-bcg').toggleClass('active');
        });
        $('#btn-view-padding').on('click', function () {
            $(this).toggleClass('active');
            $('body').toggleClass('no-padding');
        });
        $('#btn-action-select').on('click', turnOnSelect);
        $('#btn-action-swap').on('click', turnOnSwap);
        $('#btn-action-delemp').on('click', deleteEmptyGroups);
        $('#btn-action-merge').on('click', function() {
            var $target = $($('#fh5co-board .item')[0]);
            $('div.no-animate-box').appendTo($target);
            $target.children('div.fh5co-desc').html('Merged');
            deleteEmptyGroups();
        });


        /***  Selection Panel ***/
        $('#btn-selection-cancel').on('click', turnOffSelect);
        $('#btn-selection-delete').on('click', function () {
            $('a.fh5co-board-img img.selected').parent().parent().remove();
            turnOffSelect();
        });
        $('#btn-selection-inverse').on('click', function () {
            $('a.fh5co-board-img img').toggleClass('selected');
        });

        /***  Swap Panel ***/
        $('#btn-swap-cancel').on('click', turnOffSwap);


    });

}());