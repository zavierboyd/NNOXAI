loadboard = function(url){
    $('#game').load(url, clickable);
};

clickable = function(){
    $('#game table tr td').on('click', makemove);
    $('#game table tr td a').hide()
};

makemove = function(e){
    target = $(this).children('a').attr('href');
    $('#game').load(target, clickable);
};


clickmenu = function(e){
    e.preventDefault();
    $('#navbar').toggle(500);
};

hidemenu = function(){
    $('#navbar').toggle(false);
    $('#aiselection').toggle(false);
};

clickoptionbar = function(){
    $('#aiselection').toggle(500);
};

selectpony = function(e, that){
    target = $(that).html();
    ai = $(that).attr('href');
    e.preventDefault();
    $('#selector').load(ai);
    $('#aiselection').toggle(500);
};

clickplay = function(){
    $('.playbutton').on('click', playgame)
};

playgame = function(e, that){
    e.preventDefault();
    target = $(that).attr('href');
    $('#game').load(target, clickable);   //, clickplay)
};


$(document).ready(function(){
    // load the initial board to the game div
    loadboard('static/init.html')
    // load the navbar into the navbar div
    $('#navbar').load('/static/navbart.html',hidemenu);
    // load the options into the aiselection div
    $('#aiselection').load('/static/dropdownone.html');
    // when you click an option the game is played against that ai
    $('#selector').load('/static/bitzydoo.html')
});