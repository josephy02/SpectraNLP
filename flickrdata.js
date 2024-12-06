const apiKey = '6121804e178a34ebe49444858987ee5';
const apiUrl = 'https://api.flickr.com/services/rest/?method=flickr.photos.getInfo&api_key=' + apiKey + '&photo_id=10289&format=json&nojsoncallback=1';


fetch(apiUrl)
  // headers: { # don't need this for flickr
  //   'Authorization': `Bearer ${apiKey}` // Or other authentication methods
  // }
  .then(response => {
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }
    return response.json();
  })
  .then(data => {
    // Process the fetched data
    console.log(data);
  })
  .catch(error => {
    console.error('Error fetching data:', error);
  });