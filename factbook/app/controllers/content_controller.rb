class ContentController < ApplicationController
  def index
    @archive = Content.all()
  end

end
