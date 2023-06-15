var ime = null;

// Worker dobije od glavne dretve samo svoje ime
onmessage = function( e ) 
{
	ime = e.data;

	// Nakon random broja milisekundi (između 500 i 5000), pošalji natrag svoje ime, random koordinate i random kut.
    var sleep = 500 + (5000-500)*Math.random();
    setTimeout( salji, sleep );
}


salji = function()
{
	// Napravi i pošalji objekt koji se sastoji od imena, random koordianta i random kuta
	var obj = 
		{
			ime: ime, // !!! ovo radi: prvo ime je naziv svojstva objekta, a drugo je sadržaj varijable ime
			x: 900 * Math.random(),
			y: 600 * Math.random(),
			kut: 2 * Math.PI * Math.random()
		};

	postMessage( obj );

	// Ponovno, nakon random broja milisekundi (između 500 i 5000), pošalji natrag svoje ime, random koordinate i random kut.
    var sleep = 500 + (5000-500)*Math.random();
    setTimeout( salji, sleep );
}
