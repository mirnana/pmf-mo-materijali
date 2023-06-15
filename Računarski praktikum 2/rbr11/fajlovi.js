$( document ).ready( function()
{
    $("a.fileLink").on("mouseenter", dohvati_info);
} );

function dohvati_info(e) {
    let a = $(this);
    let fileName = a.prop("href");
    console.log(fileName);

    $.ajax(
        {
            url: "fajlovi.php",
            data: {fileName: fileName},
            dataType: "json",
            success: dunction(data) {
                if('greska' in data) {
                    console.log("ajax skses greska");
                }
                else {
                    let iv = $("div").html(("Datoteka: " + data.fileName + "<br>")).append("Veličina: " + data.filesize + "<br>").append("Zadnja izmjena: " + data.filemtime + "<br>").css("position", "absolute").css("top", e.clientY).css("left", e.clientX + 100);
                    $("body").append(div);
                }
            },
            error: function(xhr, status, error) {console.log("ajax error");}
        }
    )
}