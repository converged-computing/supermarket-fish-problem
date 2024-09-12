function _extends() {_extends = Object.assign || function (target) {for (var i = 1; i < arguments.length; i++) {var source = arguments[i];for (var key in source) {if (Object.prototype.hasOwnProperty.call(source, key)) {target[key] = source[key];}}}return target;};return _extends.apply(this, arguments);}class App extends React.Component {
  constructor() {
    super();
    this.state = {
      data: [],
      activeID: 0,
      imageView: false };

  }
  componentWillMount() {
    this._loadData('data.json');
  }
  componentWillUnmount() {
    this._loadData.abort();
  }
  // Fetch data, then clone it to state using destructuring
  // XHR Fallback
  _loadData(url) {
    fetch(url, {
      method: 'GET' }).

    then(response => response.json()).
    then(json => this.setState({
      data: [...json.gallery] })).

    catch(err => {
      console.log(err.message);
      try {
        const xhr = new XMLHttpRequest();
        xhr.open('GET', url);
        xhr.responseType = 'json';

        xhr.onload = () => {
          let json = xhr.response;
          this.setState({
            data: [...json.gallery] });

        };

        xhr.onerror = () => {
          throw new Error('XMLHttpRequest Failed...');
        };

        xhr.send();
      } catch (e) {
        console.log(e.message);
      }
    });
  }
  _openImageView(id) {
    this.setState({
      activeID: id,
      imageView: true });

  }
  _closeImageView() {
    this.setState({
      imageView: false });

  }
  render() {
    return /*#__PURE__*/(
      React.createElement("div", { className: "wrapper" },

      this.state.imageView ? /*#__PURE__*/
      React.createElement(ImageView, _extends({}, this.state.data[this.state.activeID], {
        _closeImageView: this._closeImageView.bind(this) })) : /*#__PURE__*/

      React.createElement(Gallery, { data: this.state.data,
        _openImageView: this._openImageView.bind(this) })));



  }}


class ImageView extends React.Component {
  render() {
    return /*#__PURE__*/(
      React.createElement("div", { className: "imageview-wrapper fadeIn" }, /*#__PURE__*/
      React.createElement("div", { className: "imageview" }, /*#__PURE__*/
      React.createElement(Image, { CSSClass: "imageview-image",
        src: this.props.src,
        alt: this.props.name }), /*#__PURE__*/
      React.createElement("div", { className: "imageview-info" }, /*#__PURE__*/
      React.createElement("button", { className: "imageview-close", onClick: this.props._closeImageView }, "x"), /*#__PURE__*/
      React.createElement("h2", null, this.props.name), /*#__PURE__*/
      React.createElement("p", null, this.props.desc), /*#__PURE__*/
      React.createElement("h3", null, "Tags"), /*#__PURE__*/
      React.createElement("ul", null,
      this.props.tags.map(tag => /*#__PURE__*/React.createElement("li", null, tag)))))));





  }}


class Gallery extends React.Component {
  render() {
    return /*#__PURE__*/(
      React.createElement("div", { className: "gallery fadeIn" },

      this.props.data.map((data) => /*#__PURE__*/
      React.createElement(Tile, { key: data.id,
        id: data.id,
        src: data.src,
        name: data.name,
        desc: data.desc,
        _openImageView: this.props._openImageView }))));
  }}


class Tile extends React.Component {
  _handleClick() {
    this.props._openImageView(this.props.id);
  }
  render() {
    return /*#__PURE__*/(
      React.createElement("div", { className: "gallery-tile", onClick: this._handleClick.bind(this) }, /*#__PURE__*/
      React.createElement("div", { className: "picture-info" }, /*#__PURE__*/
      React.createElement("h2", null, this.props.name)), /*#__PURE__*/


      React.createElement(Image, {
        CSSClass: "tile-image",
        src: this.props.src,
        alt: this.props.name })));


  }}

const Image = (props) => /*#__PURE__*/
React.createElement("img", {
  className: props.CSSClass,
  src: props.src,
  alt: props.name });

// Render app
ReactDOM.render( /*#__PURE__*/React.createElement(App, null), document.getElementById('app'));
